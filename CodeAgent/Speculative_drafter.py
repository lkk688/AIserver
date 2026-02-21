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

if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B" 
    
    print("Loading Base Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
    
    # Wrap it in our Stage 1 Drafter (Extracting at Layer 14)
    model = Stage1SpeculativeModel(base_model, exit_layer_idx=14, window_size=32)
    # Move the new drafter parameters to the correct device/dtype
    model.drafter.to(base_model.device, dtype=torch.bfloat16)
    
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