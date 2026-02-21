import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from Speculative_drafter import LocalAttentionDraftBlock, SimpleCodingDataset # Imports your saved Stage 1 classes

# ==========================================
# 1. THE ELASTIC ROUTER
# ==========================================
class ElasticComputeRouter(nn.Module):
    def __init__(self, hidden_size, num_lanes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size // 4, num_lanes, bias=False)
        )
        # Initialize with a strong safety bias towards Lane 1 (Heavy Global)
        # The model must actively learn when it is safe to pick Lane 0
        self.safety_bias = nn.Parameter(torch.tensor([-1.0, 1.0])) 

    def forward(self, hidden_state, temperature=1.0):
        logits = self.net(hidden_state) + self.safety_bias
        routed_probs = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        return routed_probs, logits

# ==========================================
# 2. THE STAGE 2 DYNAMIC MODEL
# ==========================================
class Stage2RoutedModel(nn.Module):
    def __init__(self, base_model, drafter_weights_path, exit_layer_idx=14):
        super().__init__()
        self.base_model = base_model
        self.exit_layer_idx = exit_layer_idx
        
        # 1. Load the perfectly trained Stage 1 Drafter
        self.drafter = LocalAttentionDraftBlock(base_model.config, window_size=32)
        self.drafter.load_state_dict(torch.load(drafter_weights_path))
        self.lm_head = base_model.lm_head
        
        # 2. Initialize the new Stage 2 Router
        self.router = ElasticComputeRouter(base_model.config.hidden_size, num_lanes=2)
        
        # 3. Freeze everything EXCEPT the router!
        for param in self.base_model.parameters(): 
            param.requires_grad = False
        for param in self.drafter.parameters(): 
            param.requires_grad = False

    def forward(self, input_ids, temperature=1.0):
        # Run base model to the extraction layer
        with torch.no_grad():
            outputs = self.base_model(input_ids, output_hidden_states=True)
            true_logits = outputs.logits
            intermediate_features = outputs.hidden_states[self.exit_layer_idx]
            
            # Get the Drafter's predictions
            draft_features = self.drafter(intermediate_features)
            draft_logits = self.lm_head(draft_features)
            
        # The Router analyzes the hidden features to predict Drafter success
        routed_probs, router_logits = self.router(intermediate_features, temperature=temperature)
        
        return router_logits, routed_probs, draft_logits, true_logits

# ==========================================
# 3. THE ORACLE TRAINING LOOP
# ==========================================
def train_router_dynamic(model, train_loader, val_loader, epochs=2, lr=1e-3):
    device = next(model.router.parameters()).device
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=lr)
    
    # 1. THE DTYPE FIX: Explicitly cast weights to float32
    weights = torch.tensor([1.0, 2.0], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    for epoch in range(epochs):
        model.train()
        print(f"\n--- Epoch {epoch+1} Training Elastic Router ---")
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass
            router_logits, routed_probs, draft_logits, true_logits = model(input_ids)
            
            # --- CREATE DYNAMIC GROUND TRUTH ---
            with torch.no_grad():
                draft_tokens = torch.argmax(draft_logits[:, :-1, :], dim=-1)
                true_tokens = torch.argmax(true_logits[:, :-1, :], dim=-1)
                
                # THE DTYPE FIX: Target lanes must be Long tensors for CrossEntropy
                target_lanes = torch.where(draft_tokens == true_tokens, 0, 1).view(-1).long()
                
            # THE DTYPE FIX: Cast logits to float32 for stable CrossEntropy math
            valid_router_logits = router_logits[:, :-1, :].reshape(-1, 2).float()
            
            # Calculate Loss and Backpropagate
            loss = criterion(valid_router_logits, target_lanes)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 20 == 0:
                predictions = torch.argmax(valid_router_logits, dim=-1)
                accuracy = (predictions == target_lanes).float().mean() * 100
                print(f"Batch {batch_idx} | Router Loss: {loss.item():.4f} | Routing Accuracy: {accuracy:.1f}%")

        # ==========================================
        # STAGE 2 EVALUATION PHASE
        # ==========================================
        model.eval()
        print(f"\n--- Epoch {epoch+1} Evaluation ---")
        
        total_correct = 0
        total_predictions = 0
        true_skips_predicted = 0
        total_true_skips = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                
                router_logits, routed_probs, draft_logits, true_logits = model(input_ids)
                
                draft_tokens = torch.argmax(draft_logits[:, :-1, :], dim=-1)
                true_tokens = torch.argmax(true_logits[:, :-1, :], dim=-1)
                
                target_lanes = torch.where(draft_tokens == true_tokens, 0, 1).view(-1).long()
                valid_router_logits = router_logits[:, :-1, :].reshape(-1, 2).float()
                
                predictions = torch.argmax(valid_router_logits, dim=-1)
                
                total_correct += (predictions == target_lanes).sum().item()
                total_predictions += target_lanes.numel()
                
                # Track how well it identifies "Safe to Skip" (Lane 0) scenarios
                true_skips_predicted += ((predictions == 0) & (target_lanes == 0)).sum().item()
                total_true_skips += (target_lanes == 0).sum().item()

        val_accuracy = (total_correct / total_predictions) * 100
        skip_recall = (true_skips_predicted / max(1, total_true_skips)) * 100
        
        print(f"Overall Routing Accuracy: {val_accuracy:.2f}%")
        print(f"Oracle Skip Recall (Did it catch the safe skips?): {skip_recall:.2f}%")
        print("-" * 40)

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B" 
    
    print("Loading Base Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        dtype=torch.bfloat16 # Safely use bfloat16 for weights
    )
    
    # Initialize Stage 2 Model with your saved Stage 1 weights
    model = Stage2RoutedModel(base_model, "trained_local_drafter_epoch5.pth", exit_layer_idx=14)
    model.drafter.to(base_model.device, dtype=torch.bfloat16)
    model.router.to(base_model.device, dtype=torch.bfloat16)
    
    # Load Train and Validation Datasets
    print("Loading Training Data...")
    train_dataset = SimpleCodingDataset(tokenizer, dataset_name="bigcode/the-stack-smol", data_dir="data/python", split="train", num_samples=10000)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    print("Loading Validation Data...")
    # We can use MBPP as a clean validation set
    val_dataset = SimpleCodingDataset(tokenizer, dataset_name="mbpp", split="validation", num_samples=500)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Train the dynamic oracle
    train_router_dynamic(model, train_loader, val_loader, epochs=3)
    
    # Save the trained router!
    torch.save(model.router.state_dict(), "trained_elastic_router_epoch3.pth")
    print("Stage 2 Router saved successfully!")