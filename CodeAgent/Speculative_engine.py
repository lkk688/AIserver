import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from Speculative_drafter import LocalAttentionDraftBlock
from train_stage2_router import ElasticComputeRouter

# ==========================================
# 1. THE UNIFIED ENGINE
# ==========================================
class ElasticSpeculativeEngine(nn.Module):
    def __init__(self, base_model, drafter_path, router_path, exit_layer_idx=14):
        super().__init__()
        self.base_model = base_model
        self.exit_layer_idx = exit_layer_idx
        
        # Load Drafter
        self.drafter = LocalAttentionDraftBlock(base_model.config, window_size=32)
        self.drafter.load_state_dict(torch.load(drafter_path))
        self.lm_head = base_model.lm_head
        
        # Load Router
        self.router = ElasticComputeRouter(base_model.config.hidden_size, num_lanes=2)
        self.router.load_state_dict(torch.load(router_path))
        
        # Set everything to eval mode
        self.base_model.eval()
        self.drafter.eval()
        self.router.eval()
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self):
        pass # We implement the logic directly in the generate function

# ==========================================
# 2. THE GENERATION LOOP
# ==========================================
def generate_elastic_speculative(engine, tokenizer, prompt, max_new_tokens=50, K=3):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(engine.base_model.device)
    
    start_time = time.time()
    tokens_generated = 0
    forward_steps = 0
    
    stats = {"trusted_skips": 0, "heavy_routes": 0, "drafts_generated": 0, "drafts_accepted": 0}
    
    with torch.no_grad():
        while tokens_generated < max_new_tokens:
            # 1. Run Base Model up to the Exit Layer
            outputs = engine.base_model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[engine.exit_layer_idx]
            
            # 2. Router Decision
            routed_probs, _ = engine.router(hidden_states[:, -1:, :])
            lane_choice = torch.argmax(routed_probs, dim=-1).item()
            
            # --- PATH A: ROUTER TRUSTS DRAFTER (Lane 0) ---
            if lane_choice == 0:
                stats["trusted_skips"] += 1
                draft_tokens = []
                current_input = input_ids
                
                # Drafter auto-regressively predicts K tokens
                for _ in range(K):
                    draft_out = engine.base_model(current_input, output_hidden_states=True)
                    draft_hidden = draft_out.hidden_states[engine.exit_layer_idx]
                    
                    # THE FIX: Pass the full sequence so Sliding Window Attention works!
                    d_features = engine.drafter(draft_hidden) 
                    
                    # Slice ONLY at the vocabulary projection step to save compute
                    d_logits = engine.lm_head(d_features[:, -1:, :])
                    d_token = torch.argmax(d_logits, dim=-1)
                    
                    draft_tokens.append(d_token)
                    current_input = torch.cat([current_input, d_token], dim=1)
                    stats["drafts_generated"] += 1
                
                draft_tensor = torch.cat(draft_tokens, dim=1)
                spec_input_ids = torch.cat([input_ids, draft_tensor], dim=1)
                
                # Parallel Verification (Slow Brain)
                slow_outputs = engine.base_model(spec_input_ids)
                forward_steps += 1 
                slow_logits = slow_outputs.logits
                seq_len = input_ids.shape[1]
                
                accepted_list = []
                hit_eos = False
                
                for i in range(K):
                    true_token = torch.argmax(slow_logits[:, seq_len - 1 + i, :], dim=-1).unsqueeze(1)
                    draft_tok = draft_tensor[:, i].unsqueeze(1)
                    
                    if true_token.item() == draft_tok.item():
                        accepted_list.append(draft_tok)
                        stats["drafts_accepted"] += 1
                        if draft_tok.item() == tokenizer.eos_token_id:
                            hit_eos = True
                            break
                    else:
                        accepted_list.append(true_token)
                        if true_token.item() == tokenizer.eos_token_id:
                            hit_eos = True
                        break
                
                if len(accepted_list) == K and not hit_eos:
                    bonus_token = torch.argmax(slow_logits[:, -1:, :], dim=-1)
                    accepted_list.append(bonus_token)
                    if bonus_token.item() == tokenizer.eos_token_id:
                        hit_eos = True
                        
                accepted_tensor = torch.cat(accepted_list, dim=1)
                
                # Prevent overshoot
                remaining = max_new_tokens - tokens_generated
                if accepted_tensor.shape[1] > remaining:
                    accepted_tensor = accepted_tensor[:, :remaining]
                    
                input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
                tokens_generated += accepted_tensor.shape[1]
                
                if hit_eos or input_ids[0, -1].item() == tokenizer.eos_token_id:
                    break

            # --- PATH B: ROUTER REJECTS DRAFTER (Lane 1) ---
            else:
                stats["heavy_routes"] += 1
                forward_steps += 1
                
                # Standard auto-regressive step
                next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                tokens_generated += 1
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                    
    wall_time = time.time() - start_time
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return text, tokens_generated, wall_time, forward_steps, stats

# ==========================================
# 3. BENCHMARK EXECUTION
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B" 
    
    print("Loading Base Model (Deterministic FP32)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    
    print("Loading Elastic Speculative Engine...")
    engine = ElasticSpeculativeEngine(
        base_model, 
        drafter_path="trained_local_drafter_epoch5.pth", 
        router_path="trained_elastic_router_epoch3.pth", 
        exit_layer_idx=14
    )
    engine.drafter.to(base_model.device, dtype=torch.float32)
    engine.router.to(base_model.device, dtype=torch.float32)
    
    prompt = "def calculate_fibonacci(n):\n"
    
    print(f"\nPrompt: {prompt}")
    
    # Run the Engine
    text, tokens, t_time, steps, stats = generate_elastic_speculative(
        engine, tokenizer, prompt, max_new_tokens=40, K=3
    )
    
    print(f"\nGenerated Code:\n{text}")
    print("\n" + "="*40)
    print("📊 UNIFIED ENGINE STATISTICS")
    print("="*40)
    print(f"Total Tokens Generated: {tokens}")
    print(f"Total Model Steps: {steps}")
    print(f"Router Decisions  -> Trusted Skips (Lane 0): {stats['trusted_skips']}")
    print(f"                  -> Heavy Routes  (Lane 1): {stats['heavy_routes']}")
    if stats['drafts_generated'] > 0:
        acc_rate = (stats['drafts_accepted'] / stats['drafts_generated']) * 100
        print(f"Drafter Accuracy  -> {acc_rate:.1f}% ({stats['drafts_accepted']}/{stats['drafts_generated']} accepted)")
    
    theoretical_speedup = tokens / max(1, steps)
    print(f"🔥 Theoretical Speedup: {theoretical_speedup:.2f}x fewer forward passes!")