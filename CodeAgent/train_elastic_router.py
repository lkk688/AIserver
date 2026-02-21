import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. THE ARCHITECTURE COMPONENTS
# ==========================================

class FastThinkingHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        # 1. Project to the same dimension to allow for weight tying
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU()
        )
        # 2. Final prediction layer (no bias, matching standard LLM heads)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, hidden_state):
        x = self.proj(hidden_state)
        return self.lm_head(x)

class ElasticComputeRouter(nn.Module):
    def __init__(self, hidden_size, num_lanes=3):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, hidden_size // 4, bias=False)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_size // 4, num_lanes, bias=False)
        
        # Safety Bias: Favors Lane 2 (Global) by default.
        self.safety_bias = nn.Parameter(torch.tensor([-1.0, 0.0, 2.0])) 

    def forward(self, hidden_state, temperature=1.0):
        x = self.down_proj(hidden_state)
        x = self.act(x)
        logits = self.up_proj(x) + self.safety_bias
        routed_probs = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        return routed_probs, logits

class ElasticJointLoss(nn.Module):
    def __init__(self, compute_penalties=[0.0, 0.2, 1.0], lambda_cost=0.1, alpha_fast=1.0):
        super().__init__()
        # Router Loss: Weighted to penalize missing "Global" tokens
        class_weights = torch.tensor([0.3, 1.0, 2.0], dtype=torch.float32)
        self.router_ce = nn.CrossEntropyLoss(weight=class_weights)
        
        self.fast_head_ce = nn.CrossEntropyLoss()
        
        self.lambda_cost = lambda_cost
        self.alpha_fast = alpha_fast
        self.register_buffer("penalties", torch.tensor(compute_penalties, dtype=torch.float32))

    def forward(self, router_logits, target_lanes, fast_logits, target_tokens, routed_probs):
        # 1. CAST TO FLOAT32 FOR STABILITY & DTYPE MATCHING
        router_logits = router_logits.float()
        fast_logits = fast_logits.float()
        routed_probs = routed_probs.float()
        
        # Ensure targets are Long (int64)
        target_lanes = target_lanes.long()
        target_tokens = target_tokens.long()
        
        # A. Train the Router
        loss_route = self.router_ce(router_logits, target_lanes)
        
        # B. Train the Router to save compute
        expected_cost = torch.sum(routed_probs * self.penalties, dim=-1)
        loss_cost = expected_cost.mean()
        
        # C. Train the Fast Head
        loss_fast = self.fast_head_ce(fast_logits, target_tokens)
        
        # JOINT LOSS
        total_loss = loss_route + (self.lambda_cost * loss_cost) + (self.alpha_fast * loss_fast)
        return total_loss, loss_route, loss_fast
    
# ==========================================
# 2. THE ELASTIC ATTENTION WRAPPER
# ==========================================

class ElasticQwenAttention(nn.Module):
    def __init__(self, config, base_attention_layer):
        super().__init__()
        self.config = config
        self.attn = base_attention_layer 
        
        self.router = ElasticComputeRouter(config.hidden_size, num_lanes=3)
        self.fast_brain = FastThinkingHead(config.hidden_size, config.vocab_size)
        
        # State variables to hold data for the loss function
        self.last_router_logits = None
        self.last_routed_probs = None
        self.last_fast_logits = None
        self.current_temp = 1.0 # Will be updated by the training loop

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, output_attentions=False, use_cache=False, cache_position=None, **kwargs):
        # 1. Route the tokens
        routed_probs, router_logits = self.router(hidden_states, temperature=self.current_temp)
        
        # 2. Fast Head Draft
        fast_logits = self.fast_brain(hidden_states)
        
        # Save for the training loop
        self.last_router_logits = router_logits
        self.last_routed_probs = routed_probs
        self.last_fast_logits = fast_logits

        # 3. Execute Lanes
        out_skip = hidden_states 
        
        attn_outputs = self.attn(
            hidden_states=hidden_states, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs
        )
        out_global = attn_outputs[0] 
        out_local = out_global * 0.9 

        # 4. Blend outputs differentiably (WITH OVERRIDE SWITCH)
        # If training, OR if we explicitly force the Slow Brain to be accurate during verification
        if self.training or getattr(self, "force_global", False):
            final_output = out_global
        else:
            p_skip = routed_probs[:, :, 0].unsqueeze(-1)
            p_local = routed_probs[:, :, 1].unsqueeze(-1)
            p_global = routed_probs[:, :, 2].unsqueeze(-1)
            final_output = (p_skip * out_skip) + (p_local * out_local) + (p_global * out_global)

        if len(attn_outputs) > 1:
            return (final_output,) + attn_outputs[1:]
        return (final_output,)

# ==========================================
# 3. MODEL INJECTION
# ==========================================

def inject_elastic_routers(model):
    print("Injecting Elastic Routers and Fast Heads into model layers...")
    for i, layer in enumerate(model.model.layers):
        base_attn = layer.self_attn
        config = model.config
        
        target_device = next(base_attn.parameters()).device
        target_dtype = next(base_attn.parameters()).dtype
        
        # 1. Create the new wrapper
        elastic_attn = ElasticQwenAttention(config, base_attn)
        
        # --- THE UPGRADE: TIE EMBEDDINGS ---
        # 2. Tie the Fast Head's prediction layer to the base model's vocabulary weights
        # We do this BEFORE we move it to the GPU to ensure seamless pointer sharing
        elastic_attn.fast_brain.lm_head.weight = model.lm_head.weight
        
        # 3. Move the wrapper (and its new router/fast head) to match the base layer
        elastic_attn = elastic_attn.to(device=target_device, dtype=target_dtype)
        
        # 4. Hot-swap the attention mechanism
        layer.self_attn = elastic_attn
        
    print(f"Successfully injected and weight-tied into {len(model.model.layers)} layers.")
    return model

# ==========================================
# 4. TRAINING & EVALUATION LOOPS
# ==========================================

def set_global_mode(model, mode=True):
    """Forces all layers to use pure Global Attention for pristine verification."""
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'force_global'):
            layer.self_attn.force_global = mode
            
def benchmark_generation(model, tokenizer, prompt, max_new_tokens=50, trust_threshold=0.85):
    """
    Runs dynamic generation and calculates real acceptance rates.
    """
    print(f"\n--- Running Dynamic Inference Benchmark ---")
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    total_compute_cost = 0.0
    tokens_generated = 0
    
    # Tracking for the paper's metrics
    stats = {"trusted_skips": 0, "drafts_accepted": 0, "drafts_rejected": 0, "global_passes": 0}
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1. Get the current hidden states
            outputs = model(input_ids, output_hidden_states=True)
            # Use the hidden state from an early layer (e.g., layer 2) for the router
            early_hidden = outputs.hidden_states[2] 
            
            # For simplicity in this benchmark, we'll ask the FIRST layer's router what to do
            # In a full setup, you might aggregate router votes or use a designated router layer
            first_layer_attn = model.model.layers[0].self_attn
            
            routed_probs, _ = first_layer_attn.router(early_hidden[:, -1:, :])
            fast_logits = first_layer_attn.fast_brain(early_hidden[:, -1:, :])
            
            lane_choice = torch.argmax(routed_probs, dim=-1).item()
            router_conf = torch.max(routed_probs, dim=-1)[0].item()
            
            fast_token = torch.argmax(fast_logits, dim=-1)
            fast_conf = torch.softmax(fast_logits, dim=-1).max().item()
            
            # --- PATH A: TRUSTED SKIP ---
            if lane_choice == 0 and router_conf > trust_threshold and fast_conf > trust_threshold:
                input_ids = torch.cat([input_ids, fast_token], dim=1)
                total_compute_cost += 0.0 # 100% savings
                stats["trusted_skips"] += 1
                
            # --- PATH B: SPECULATIVE DRAFT & VERIFY ---
            elif lane_choice == 0:
                # We draft 1 token for this simple benchmark (can be expanded to K drafts)
                draft_token = fast_token
                
                # Verify using the Slow Brain (the 'outputs' we already calculated)
                slow_next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
                
                if draft_token.item() == slow_next_token.item():
                    # Accept!
                    input_ids = torch.cat([input_ids, draft_token], dim=1)
                    stats["drafts_accepted"] += 1
                else:
                    # Reject & Override
                    input_ids = torch.cat([input_ids, slow_next_token], dim=1)
                    stats["drafts_rejected"] += 1
                
                total_compute_cost += 1.0 
                
            # --- PATH C: GLOBAL ---
            else:
                slow_next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
                input_ids = torch.cat([input_ids, slow_next_token], dim=1)
                
                # Cost is 0.5 for local (Lane 1), 1.0 for global (Lane 2)
                cost = 0.5 if lane_choice == 1 else 1.0
                total_compute_cost += cost
                stats["global_passes"] += 1
                
            tokens_generated += 1
            
            # Stop if EOS token is generated
            if input_ids[0, -1].item() == tokenizer.eos_token_id:
                break

    # Calculate final metrics
    avg_cost_per_token = total_compute_cost / tokens_generated
    actual_savings = (1.0 - avg_cost_per_token) * 100
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    print(f"Prompt: {prompt}")
    print(f"Generated Code:\n{generated_text}\n")
    print("--- Performance Metrics ---")
    print(f"Tokens Generated: {tokens_generated}")
    print(f"Trusted Skips (0 cost): {stats['trusted_skips']}")
    print(f"Drafts Accepted: {stats['drafts_accepted']}")
    print(f"Drafts Rejected: {stats['drafts_rejected']}")
    print(f"Global Passes: {stats['global_passes']}")
    print(f"Actual Compute Savings: {actual_savings:.1f}%")


def full_speculative_benchmark(model, tokenizer, prompt, max_new_tokens=50, num_drafts=3, trust_threshold=0.85):
    print(f"\n--- Running Full Speculative Inference (K={num_drafts}) ---")
    model.eval()
    
    # THE CRITICAL FIX: The base model must always act as a flawless referee.
    # We do not let internal routers corrupt the base model's KV cache or hidden states.
    set_global_mode(model, True) 
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    stats = {"trusted_skips": 0, "drafts_accepted": 0, "drafts_rejected": 0, "global_passes": 0}
    tokens_generated = 0
    
    EXIT_LAYER_IDX = 10 
    router_layer = model.model.layers[EXIT_LAYER_IDX].self_attn

    with torch.no_grad():
        while tokens_generated < max_new_tokens:
            # 1. Run the pristine model to get the context up to our Exit Layer
            # (In production C++, this stops executing at Layer 10 to save compute)
            outputs = model(input_ids, output_hidden_states=True)
            deep_hidden = outputs.hidden_states[EXIT_LAYER_IDX] 
            
            # 2. Ask the Router and Fast Head at Layer 10
            routed_probs, _ = router_layer.router(deep_hidden[:, -1:, :])
            lane_choice = torch.argmax(routed_probs, dim=-1).item()
            router_conf = torch.max(routed_probs, dim=-1)[0].item()
            
            fast_logits = router_layer.fast_brain(deep_hidden[:, -1:, :])
            fast_token = torch.argmax(fast_logits, dim=-1)
            fast_conf = torch.softmax(fast_logits, dim=-1).max().item()
            
            # ==========================================
            # PATH A: THE TRUSTED SKIP
            # ==========================================
            if lane_choice == 0 and router_conf > trust_threshold and fast_conf > trust_threshold:
                input_ids = torch.cat([input_ids, fast_token], dim=1)
                stats["trusted_skips"] += 1
                tokens_generated += 1

            # ==========================================
            # PATH B: SPECULATIVE DRAFT & VERIFY
            # ==========================================
            elif lane_choice == 0:
                draft_tokens = []
                current_draft_input = input_ids
                
                # Fast Head autoregressively drafts K tokens
                for _ in range(num_drafts):
                    draft_out = model(current_draft_input, output_hidden_states=True)
                    draft_hidden = draft_out.hidden_states[EXIT_LAYER_IDX]
                    
                    f_logits = router_layer.fast_brain(draft_hidden[:, -1:, :])
                    f_token = torch.argmax(f_logits, dim=-1)
                    
                    draft_tokens.append(f_token)
                    current_draft_input = torch.cat([current_draft_input, f_token], dim=1)
                
                draft_tensor = torch.cat(draft_tokens, dim=1) 
                speculative_input = torch.cat([input_ids, draft_tensor], dim=1)
                
                # The Pristine Slow Brain verifies the drafts
                slow_outputs = model(speculative_input)
                slow_logits = slow_outputs.logits
                seq_len = input_ids.shape[1]
                accepted_list = []
                
                for i in range(num_drafts):
                    true_token = torch.argmax(slow_logits[:, seq_len - 1 + i, :], dim=-1).unsqueeze(1)
                    draft_token = draft_tensor[:, i].unsqueeze(1)
                    
                    if true_token.item() == draft_token.item():
                        accepted_list.append(draft_token)
                        stats["drafts_accepted"] += 1
                    else:
                        accepted_list.append(true_token)
                        stats["drafts_rejected"] += 1
                        break
                
                if len(accepted_list) == num_drafts:
                    bonus_token = torch.argmax(slow_logits[:, -1:, :], dim=-1)
                    accepted_list.append(bonus_token)
                    
                accepted_tensor = torch.cat(accepted_list, dim=1)
                input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
                tokens_generated += accepted_tensor.shape[1]

            # ==========================================
            # PATH C: HEAVY GLOBAL LOGIC
            # ==========================================
            else:
                slow_outputs = model(input_ids)
                next_token = torch.argmax(slow_outputs.logits[:, -1:, :], dim=-1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                stats["global_passes"] += 1
                tokens_generated += 1
            
            if input_ids[0, -1].item() == tokenizer.eos_token_id:
                break

    #print(f"Generated Code:\n{tokenizer.decode(input_ids[0], skip_special_tokens=True)}\n")
    print(f"Generated Code:\n{tokenizer.decode(input_ids[0], skip_special_tokens=True)}\n")
    print("--- Benchmark Stats ---")
    print(f"Total Tokens Generated: {tokens_generated}")
    print(f"Global Passes (System 2): {stats['global_passes']}")
    print(f"Trusted Skips (System 1): {stats['trusted_skips']}")
    print(f"Speculative Drafts Accepted: {stats['drafts_accepted']}")
    print(f"Speculative Drafts Rejected: {stats['drafts_rejected']}")
    
def train_and_evaluate(model, train_loader, val_loader, epochs=1, lr=1e-3):
    # FREEZE entire model
    for param in model.parameters():
        param.requires_grad = False

    # UNFREEZE only the specific Router and Fast Head projection parameters
    trainable_params = []
    for name, module in model.named_modules():
        if isinstance(module, ElasticComputeRouter):
            for param in module.parameters():
                param.requires_grad = True
                trainable_params.append(param)
                
        elif isinstance(module, FastThinkingHead):
            for param_name, param in module.named_parameters():
                # THE FIX: Do not unfreeze the tied vocabulary weights!
                if "lm_head" not in param_name:
                    param.requires_grad = True
                    trainable_params.append(param)

    device = model.device
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    criterion = ElasticJointLoss(lambda_cost=0.15, alpha_fast=1.0).to(device)
    
    start_temp = 2.0
    min_temp = 0.5

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        current_temp = max(min_temp, start_temp * (0.75 ** epoch))
        
        print(f"\n--- Epoch {epoch+1} Training | Temp: {current_temp:.2f} ---")
        
        # Set temperature for all routers
        for layer in model.model.layers:
            layer.self_attn.current_temp = current_temp

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            target_lanes = batch['router_labels'].to(device).view(-1).long()
            
            target_tokens = torch.roll(input_ids, shifts=-1, dims=1)
            target_tokens[:, -1] = -100 
            target_tokens = target_tokens.view(-1).long()
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            
            batch_loss = 0
            batch_r_loss = 0
            batch_f_loss = 0
            num_layers = len(model.model.layers)
            
            # --- THE OOM FIX: PER-LAYER LOSS CALCULATION ---
            for layer in model.model.layers:
                # Extract this layer's specific logits
                r_logits = layer.self_attn.last_router_logits.view(-1, 3)
                probs = layer.self_attn.last_routed_probs.view(-1, 3)
                
                # Fast Head logits for THIS layer only: shape (batch * seq_len, vocab_size)
                f_logits = layer.self_attn.last_fast_logits.view(-1, model.config.vocab_size)
                
                # Calculate Joint Loss for this specific layer
                loss, r_loss, f_loss = criterion(r_logits, target_lanes, f_logits, target_tokens, probs)
                
                # Scale the loss because we are summing gradients across 28 layers
                scaled_loss = loss / num_layers
                
                # Calculate gradients and IMMEDIATELY free the graph memory
                scaled_loss.backward()
                
                batch_loss += scaled_loss.item()
                batch_r_loss += (r_loss.item() / num_layers)
                batch_f_loss += (f_loss.item() / num_layers)
                
                # Manually delete the massive tensors to keep VRAM clean
                del r_logits, probs, f_logits, loss, r_loss, f_loss
                
            # Step the optimizer once all 28 layers have accumulated their gradients
            optimizer.step()
            total_loss += batch_loss
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} | Total Loss: {batch_loss:.4f} | Route Loss: {batch_r_loss:.4f} | Fast Head Loss: {batch_f_loss:.4f}")
        
        
        # --- EVALUATION ---
        print(f"\n--- Epoch {epoch+1} Evaluation ---")
        model.eval()
        lane_counts = {0: 0, 1: 0, 2: 0}
        total_tokens = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                outputs = model(input_ids=input_ids)
                
                for layer in model.model.layers:
                    probs = layer.self_attn.last_routed_probs
                    choices = torch.argmax(probs, dim=-1).view(-1)
                    
                    for lane in range(3):
                        lane_counts[lane] += (choices == lane).sum().item()
                    total_tokens += choices.numel()

        pct_0 = lane_counts[0] / total_tokens
        pct_1 = lane_counts[1] / total_tokens
        pct_2 = lane_counts[2] / total_tokens
        
        compute_cost = (pct_0 * 0.0) + (pct_1 * 0.2) + (pct_2 * 1.0)
        compute_savings = 1.0 - compute_cost
        
        print("Routing Distribution:")
        print(f"  Lane 0 (Skip/Fast):   {pct_0*100:.1f}%")
        print(f"  Lane 1 (Local/SWA):   {pct_1*100:.1f}%")
        print(f"  Lane 2 (Global/Heavy):{pct_2*100:.1f}%")
        print(f"Estimated Compute Savings: {compute_savings*100:.1f}%")
        print("-" * 30)

# ==========================================
# 5. DATASET & EXECUTION 
# ==========================================

class HuggingFaceCodingDataset(Dataset):
    def __init__(self, tokenizer, dataset_name="mbpp", split="train", num_samples=500, seq_len=128):
        print(f"Downloading {dataset_name} from Hugging Face...")
        hf_dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        
        self.input_ids = []
        self.labels = []
        
        print("Tokenizing and generating heuristic routing labels...")
        id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
        fast_keywords = ['\n', 'Ġdef', 'Ġimport', 'Ġreturn', '(', ')', ':', ',', 'Ġ=', 'Ġ', 'Ġnp', 'Ġself', 'Ġin', 'Ġfor']
        
        samples_processed = 0
        for item in tqdm(hf_dataset):
            if samples_processed >= num_samples:
                break
                
            code_text = item['code'] 
            tokens = tokenizer(code_text, truncation=True, max_length=seq_len, return_tensors="pt")
            
            if tokens.input_ids.shape[1] < seq_len:
                continue
                
            sample_input_ids = tokens.input_ids[0]
            sample_labels = []
            history = []
            
            for i, token_id in enumerate(sample_input_ids):
                token_id_int = token_id.item()
                token_str = id_to_token.get(token_id_int, "")
                
                if any(fast in token_str for fast in fast_keywords) or len(token_str) <= 2:
                    sample_labels.append(0)
                elif token_id_int in history[-50:]: 
                    sample_labels.append(1)
                else:
                    sample_labels.append(2)
                history.append(token_id_int)
            
            self.input_ids.append(sample_input_ids)
            self.labels.append(torch.tensor(sample_labels))
            samples_processed += 1

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return {"input_ids": self.input_ids[idx], "router_labels": self.labels[idx]}

if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B" 
    
    print("Loading Base Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
    
    model = inject_elastic_routers(model)
    
    train_dataset = HuggingFaceCodingDataset(tokenizer, dataset_name="bigcode/the-stack-smol", split="train", num_samples=20000)
    val_dataset = HuggingFaceCodingDataset(tokenizer, dataset_name="bigcode/the-stack-smol", split="validation", num_samples=100)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    train_and_evaluate(model, train_loader, val_loader, epochs=3)
    
    benchmark_generation(model, tokenizer, prompt="def calculate_fibonacci(n):")
    # ==========================================
    # 4. RUN THE SPECULATIVE BENCHMARK
    # ==========================================
    test_prompt = "def calculate_fibonacci(n):\n"
    
    # We test it with K=3 drafts and a high trust threshold to ensure accuracy
    full_speculative_benchmark(
        model=model, 
        tokenizer=tokenizer, 
        prompt=test_prompt, 
        max_new_tokens=40, 
        num_drafts=3, 
        trust_threshold=0.85
    )
