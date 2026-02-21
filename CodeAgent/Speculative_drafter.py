import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

# ==========================================
# 1. THE LOCAL ATTENTION DRAFTER (STAGE 1)
# ==========================================

class LocalAttentionDraftBlock(nn.Module):
    def __init__(self, config, window_size=32):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.window_size = window_size
        
        # Lightweight Single-Head Attention
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # The Adapter mapping back to vocabulary space
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Linear(self.hidden_size // 2, self.hidden_size)
        )

    def forward(self, hidden_states):
        seq_len = hidden_states.size(1)
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Sliding Window Causal Mask
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device)
        mask = torch.tril(mask) # Left-to-Right
        window_mask = torch.triu(torch.ones_like(mask), diagonal=-self.window_size + 1)
        local_mask = mask & window_mask
        
        attn_bias = torch.zeros_like(local_mask, dtype=hidden_states.dtype)
        attn_bias.masked_fill_(~local_mask, float('-inf'))
        
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        attn_weights = attn_weights + attn_bias
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = self.o_proj(attn_output)
        
        draft_features = hidden_states + attn_output
        final_features = draft_features + self.adapter(draft_features)
        
        return final_features

class Stage1SpeculativeModel(nn.Module):
    def __init__(self, base_model, exit_layer_idx=14, window_size=32):
        super().__init__()
        self.base_model = base_model 
        self.exit_layer_idx = exit_layer_idx
        
        self.drafter = LocalAttentionDraftBlock(base_model.config, window_size=window_size)
        self.lm_head = base_model.lm_head # Tie weights natively
        
        # Freeze the base model completely
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        # 1. Run base model to get the target layer's features
        with torch.no_grad():
            outputs = self.base_model(input_ids, output_hidden_states=True)
            
        intermediate_features = outputs.hidden_states[self.exit_layer_idx]
        
        # 2. Draft using Local Attention
        draft_features = self.drafter(intermediate_features)
        
        # 3. Project to Vocab
        draft_logits = self.lm_head(draft_features)
        
        return draft_logits

# ==========================================
# 2. SIMPLIFIED TRAINING LOOP
# ==========================================

def train_drafter(model, train_loader, val_loader, epochs=3, lr=1e-3):
    device = next(model.parameters()).device
    
    # Only the drafter block requires gradients!
    optimizer = torch.optim.AdamW(model.drafter.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        print(f"\n--- Epoch {epoch+1} Training Drafter ---")
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            
            # Forward Pass
            draft_logits = model(input_ids)
            
            # Prepare targets (shift left by 1 to predict the NEXT token)
            shift_logits = draft_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            # Calculate standard language modeling loss
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} | Drafter CE Loss: {loss.item():.4f}")

        # --- EVALUATION (DRAFTING ACCURACY) ---
        print(f"\n--- Epoch {epoch+1} Evaluation ---")
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                
                # We need the base model's true outputs to act as the Ground Truth
                true_outputs = model.base_model(input_ids)
                true_tokens = torch.argmax(true_outputs.logits[:, :-1, :], dim=-1)
                
                # Get drafter's predictions
                draft_logits = model(input_ids)
                draft_tokens = torch.argmax(draft_logits[:, :-1, :], dim=-1)
                
                # How often does the Drafter perfectly guess the Slow Brain's output?
                correct_predictions += (draft_tokens == true_tokens).sum().item()
                total_predictions += true_tokens.numel()

        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Drafter Top-1 Accuracy vs Base Model: {accuracy:.2f}%")
        print("-" * 30)

# ==========================================
# 3. DATASET & EXECUTION
# ==========================================

class SimpleCodingDataset(Dataset):
    def __init__(self, tokenizer, dataset_name="mbpp", split="train", num_samples=2000, seq_len=128, data_dir=None):
        print(f"Downloading {dataset_name}...")
        # Added data_dir parameter to isolate specific languages
        hf_dataset = load_dataset(dataset_name, data_dir=data_dir, split=split)
        
        self.input_ids = []
        print("Tokenizing data...")
        
        samples_processed = 0
        for item in tqdm(hf_dataset):
            if samples_processed >= num_samples:
                break
                
            # Safely check multiple common column names without crashing
            code_text = item.get('content') or item.get('code') or item.get('text')
            
            if not code_text:
                continue # Skip if empty
                
            tokens = tokenizer(code_text, truncation=True, max_length=seq_len, return_tensors="pt")
            
            if tokens.input_ids.shape[1] >= seq_len:
                self.input_ids.append(tokens.input_ids[0])
                samples_processed += 1

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, idx): return {"input_ids": self.input_ids[idx]}

import time
import torch

def generate_baseline(model, tokenizer, prompt, max_new_tokens=50):
    """Standard Autoregressive Generation (The Baseline)"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    start_time = time.time()
    tokens_generated = 0
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token = torch.argmax(outputs.logits[:, -1:, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            tokens_generated += 1
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    wall_time = time.time() - start_time
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return text, tokens_generated, wall_time, tokens_generated # Steps = tokens generated


def generate_speculative(stage1_model, tokenizer, prompt, max_new_tokens=50, K=3):
    """Single-Model Speculative Generation using the trained Drafter"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(stage1_model.base_model.device)
    
    start_time = time.time()
    tokens_generated = 0
    forward_steps = 0
    
    stats = {"drafts_generated": 0, "drafts_accepted": 0}
    
    with torch.no_grad():
        while tokens_generated < max_new_tokens:
            draft_tokens = []
            current_input = input_ids
            
            # --- PHASE 1: DRAFTING ---
            for _ in range(K):
                # 1. Run base model to the intermediate layer (e.g., Layer 14)
                # In a production kernel, we would STOP execution at Layer 14 here to save 50% compute
                outputs = stage1_model.base_model(current_input, output_hidden_states=True)
                hidden_states = outputs.hidden_states[stage1_model.exit_layer_idx]
                
                # 2. Drafter predicts the next token
                draft_logits = stage1_model.drafter(hidden_states)
                next_draft_token = torch.argmax(stage1_model.lm_head(draft_logits)[:, -1:, :], dim=-1)
                
                draft_tokens.append(next_draft_token)
                current_input = torch.cat([current_input, next_draft_token], dim=1)
                stats["drafts_generated"] += 1
            
            draft_tensor = torch.cat(draft_tokens, dim=1)
            spec_input_ids = torch.cat([input_ids, draft_tensor], dim=1)
            
            # --- PHASE 2: VERIFICATION (Inside generate_speculative) ---
            slow_outputs = stage1_model.base_model(spec_input_ids)
            forward_steps += 1 
            
            slow_logits = slow_outputs.logits
            seq_len = input_ids.shape[1]
            accepted_list = []
            hit_eos = False # NEW: Track if we hit EOS during verification
            
            # Cascading Verification
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
            
            # Bonus token if all drafts were perfect and no EOS hit
            if len(accepted_list) == K and not hit_eos:
                bonus_token = torch.argmax(slow_logits[:, -1:, :], dim=-1)
                accepted_list.append(bonus_token)
                if bonus_token.item() == tokenizer.eos_token_id:
                    hit_eos = True
                    
            accepted_tensor = torch.cat(accepted_list, dim=1)
            
            # THE OVERSHOOT FIX: Truncate if we generated too many tokens
            remaining_tokens = max_new_tokens - tokens_generated
            if accepted_tensor.shape[1] > remaining_tokens:
                accepted_tensor = accepted_tensor[:, :remaining_tokens]
                
            input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
            tokens_generated += accepted_tensor.shape[1]
            
            if hit_eos or input_ids[0, -1].item() == tokenizer.eos_token_id:
                break
                
    wall_time = time.time() - start_time
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return text, tokens_generated, wall_time, forward_steps, stats

def run_evaluation_suite(stage1_model, tokenizer, test_prompts):
    print("\n" + "="*50)
    print("🚀 STAGE 1: SPECULATIVE DECODING BENCHMARK")
    print("="*50)
    
    stage1_model.eval()
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n[Prompt {i+1}]: {prompt}")
        
        # 1. Run Baseline
        base_text, base_tokens, base_time, base_steps = generate_baseline(
            stage1_model.base_model, tokenizer, prompt, max_new_tokens=40
        )
        
        # 2. Run Speculative
        spec_text, spec_tokens, spec_time, spec_steps, stats = generate_speculative(
            stage1_model, tokenizer, prompt, max_new_tokens=40, K=3
        )
        
        # 3. Accuracy Check (Crucial!)
        is_exact_match = (base_text == spec_text)
        
        # 4. Metrics Calculation
        acc_rate = (stats['drafts_accepted'] / max(1, stats['drafts_generated'])) * 100
        step_speedup = base_steps / max(1, spec_steps)
        
        print(f"  ✅ Exact Match Verified: {is_exact_match}")
        if not is_exact_match:
            print(f"     [!] WARNING: Speculative decoding altered the base model's logic!")
            print(f"     --- BASELINE ---\n{base_text}")
            print(f"     --- SPECULATIVE ---\n{spec_text}")
            
        print(f"  🎯 Drafter Acceptance Rate: {acc_rate:.1f}% ({stats['drafts_accepted']}/{stats['drafts_generated']})")
        print(f"  ⚡ Model Steps Required: {spec_steps} (Baseline required {base_steps})")
        print(f"  🔥 Theoretical Speedup: {step_speedup:.2f}x fewer forward passes!")
        print(f"  ⏱️  Wall-clock Time: {spec_time:.2f}s (Baseline: {base_time:.2f}s)")
        print("-" * 40)

def main_train(model, tokenizer):
    # We use a pure language modeling dataset now (no heuristic labels needed!)
    # I increased the samples to 2000 for better learning. Use "mbpp" "bigcode/the-stack-smol" for real training.
    #train_dataset = SimpleCodingDataset(tokenizer, dataset_name="bigcode/the-stack-smol", split="train", num_samples=2000)
    #val_dataset = SimpleCodingDataset(tokenizer, dataset_name="bigcode/the-stack-smol", split="validation", num_samples=100)
    
    train_dataset = SimpleCodingDataset(
        tokenizer, 
        dataset_name="bigcode/the-stack-smol", 
        data_dir="data/python", # ISOLATE PYTHON
        split="train", 
        num_samples=20000
    )
    
    val_dataset = SimpleCodingDataset(
        tokenizer, 
        dataset_name="mbpp", 
        split="validation", 
        num_samples=100
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    train_drafter(model, train_loader, val_loader, epochs=5)
    
    # Save the trained drafter weights
    torch.save(model.drafter.state_dict(), "trained_local_drafter_epoch5.pth")
    print("Stage 1 Drafter saved successfully!")
    
if __name__ == "__main__":
    # MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B" 
    
    # print("Loading Base Model and Tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
    
    # # Wrap it in our Stage 1 Drafter (Extracting at Layer 14)
    # model = Stage1SpeculativeModel(base_model, exit_layer_idx=14, window_size=32)
    # # Move the new drafter parameters to the correct device/dtype
    # #model.drafter.to(base_model.device, dtype=torch.bfloat16)

    # #main_train(model, tokenizer)
    # model = model.drafter.load_state_dict(torch.load("trained_local_drafter_epoch5.pth"))
    # model.drafter.to(base_model.device, dtype=torch.bfloat16)
    # test_prompts = [
    #     "def fibonacci(n):",
    #     "class DataLoader:\n    def __init__(self, dataset):",
    #     "import numpy as np\nimport pandas as pd\n"
    # ]
    
    # run_evaluation_suite(model, tokenizer, test_prompts)

    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B" 
    
    print("Loading Base Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16, attn_implementation="eager")
    
    # 1. Initialize the wrapper class (This creates the 'model' variable)
    model = Stage1SpeculativeModel(base_model, exit_layer_idx=14, window_size=32)
    
    # 2. THE FIX: Just call the load function directly. 
    # Do NOT put "model = " at the start of this line!
    model.drafter.load_state_dict(torch.load("trained_local_drafter_epoch5.pth"))
    
    # 3. Move the drafter to the GPU
    model.drafter.to(base_model.device, dtype=torch.bfloat16)
    
    # --- THE FIX: Eliminate hardware rounding drift ---
    print("\nUpcasting model to FP32 to guarantee deterministic exact-match...")
    model.to(torch.float32)
    
    # 4. Run the Evaluation Suite
    test_prompts = [
        "def fibonacci(n):",
        "class DataLoader:\n    def __init__(self, dataset):",
        "import numpy as np\nimport pandas as pd\n"
    ]
    
    run_evaluation_suite(model, tokenizer, test_prompts)