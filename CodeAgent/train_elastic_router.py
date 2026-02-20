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
        # A lightweight 2-layer MLP to guess the next token instantly
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, vocab_size)
        )
        
    def forward(self, hidden_state):
        return self.net(hidden_state)

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

        # 4. Blend outputs
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
        
        elastic_attn = ElasticQwenAttention(config, base_attn)
        # Move both the router and the fast brain to the correct device/dtype
        elastic_attn = elastic_attn.to(device=target_device, dtype=target_dtype)
        
        layer.self_attn = elastic_attn
        
    print(f"Successfully injected into {len(model.model.layers)} layers.")
    return model

# ==========================================
# 4. TRAINING & EVALUATION LOOPS
# ==========================================

def train_and_evaluate(model, train_loader, val_loader, epochs=1, lr=1e-3):
    # FREEZE entire model
    for param in model.parameters():
        param.requires_grad = False

    # UNFREEZE the Router and the Fast Head
    trainable_params = []
    for name, module in model.named_modules():
        if isinstance(module, ElasticComputeRouter) or isinstance(module, FastThinkingHead):
            for param in module.parameters():
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
            target_lanes = batch['router_labels'].to(device).view(-1) 
            
            # To train the Fast Head, it needs to predict the *next* token.
            # We shift the input_ids to the left to create the target tokens.
            # Shape: (batch, seq_len)
            target_tokens = torch.roll(input_ids, shifts=-1, dims=1)
            # The last token has no "next" target, so we ignore it (set to -100 for CE loss)
            target_tokens[:, -1] = -100 
            target_tokens = target_tokens.view(-1)
            
            # Forward pass
            outputs = model(input_ids=input_ids)
            
            # Extract routing and fast-head data
            all_r_logits, all_probs, all_f_logits = [], [], []
            for layer in model.model.layers:
                all_r_logits.append(layer.self_attn.last_router_logits.view(-1, 3))
                all_probs.append(layer.self_attn.last_routed_probs.view(-1, 3))
                # Fast Head logits: shape (batch * seq_len, vocab_size)
                all_f_logits.append(layer.self_attn.last_fast_logits.view(-1, model.config.vocab_size))
                
            all_r_logits = torch.cat(all_r_logits, dim=0)
            all_probs = torch.cat(all_probs, dim=0)
            all_f_logits = torch.cat(all_f_logits, dim=0)
            
            # Expand targets to match the number of layers
            num_layers = len(model.model.layers)
            expanded_lanes = target_lanes.repeat(num_layers)
            expanded_tokens = target_tokens.repeat(num_layers)
            
            # Calculate Joint Loss
            loss, r_loss, f_loss = criterion(all_r_logits, expanded_lanes, all_f_logits, expanded_tokens, all_probs)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} | Total Loss: {loss.item():.4f} | Route Loss: {r_loss.item():.4f} | Fast Head Loss: {f_loss.item():.4f}")

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
    
    train_dataset = HuggingFaceCodingDataset(tokenizer, split="train", num_samples=500)
    val_dataset = HuggingFaceCodingDataset(tokenizer, split="validation", num_samples=50)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    train_and_evaluate(model, train_loader, val_loader, epochs=3)