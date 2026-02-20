import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

#pip install accelerate

# --- CONFIGURATION ---
# For research, use a smaller proxy if the 80B model is too heavy for local analysis
# Or use the full model if you have 2x A100s.
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct" 
PROMPT_CODE = """
def calculate_moving_average(data, window_size):
    import numpy as np
    if not data:
        return []
    
    # Calculate cumulative sum
    cumsum = np.cumsum(np.insert(data, 0, 0))
    
    # Return the moving average
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size
"""

def run_attention_experiment():
    print(f"Loading {MODEL_ID} for inspection... (this may take a minute)")
    
    # 1. Load Model with 'output_attentions=True'
    # This is the key switch that vLLM lacks easily.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager", # 'sdpa' or 'flash_attention_2' often hide weights too
        output_attentions=True 
    )

    # 2. Tokenize and Generate
    inputs = tokenizer(PROMPT_CODE, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs.attentions is a tuple of shape (num_layers, batch, heads, seq_len, seq_len)
    # We want the LAST layer usually (Layer -1), as it represents the final aggregation.
    # But for Qwen/Hybrid models, checking the middle layers is also valuable.
    
    # Let's analyze the LAST layer
    last_layer_attn = outputs.attentions[-1] # Shape: (1, Num_Heads, Seq_Len, Seq_Len)
    
    # Average across all heads to get a "General Attention" view
    # Shape becomes: (Seq_Len, Seq_Len)
    avg_attn = last_layer_attn[0].mean(dim=0).float().cpu().numpy()
    
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    # Clean up token strings for plotting (remove the special GGL character)
    clean_tokens = [t.replace('Ġ', '').replace('Ċ', '\\n') for t in tokens]

    # --- ANALYSIS: CALCULATE "SINK-ADJUSTED LOCAL MASS" ---
    WINDOW_SIZE = 10
    SINK_SIZE = 4  # The first 4 tokens act as the "trash can" for attention
    locality_scores = []

    for i in range(len(tokens)):
        # Get attention distribution for token 'i' (what token 'i' attended to)
        attn_dist = avg_attn[i, :i+1] 
        
        # If we are too early in the sequence, just default to 1.0 (highly local)
        if i < WINDOW_SIZE + SINK_SIZE:
            local_mass = 1.0
        else:
            # 1. Calculate how much attention was sucked up by the sink
            sink_mass = np.sum(attn_dist[:SINK_SIZE])
            
            # 2. Calculate the remaining "meaningful" attention
            meaningful_mass = 1.0 - sink_mass
            
            # 3. Calculate raw local attention
            raw_local_mass = np.sum(attn_dist[-WINDOW_SIZE:])
            
            # 4. Normalize the local attention against the meaningful attention
            if meaningful_mass < 1e-5:
                # Edge case: If the sink literally absorbed 100% of the attention
                local_mass = 0.0 
            else:
                local_mass = raw_local_mass / meaningful_mass
                
            # Cap at 1.0 just in case floating point math gets weird
            local_mass = min(1.0, local_mass)
        
        locality_scores.append(local_mass)

    # --- VISUALIZATION ---
    print("Generating Heatmap...")
    plot_results(clean_tokens, locality_scores, avg_attn)

def plot_results(tokens, locality_scores, attn_matrix):
    plt.figure(figsize=(20, 12))
    
    # Subplot 1: The "Locality Score" Bar Chart
    plt.subplot(2, 1, 1)
    x = range(len(tokens))
    colors = ['red' if s < 0.5 else 'skyblue' for s in locality_scores]
    plt.bar(x, locality_scores, color=colors)
    plt.axhline(y=0.5, color='gray', linestyle='--', label="Global Attention Threshold")
    
    # Annotate significant drops (Global lookups)
    for i, score in enumerate(locality_scores):
        if score < 0.5:
            plt.text(i, score + 0.05, tokens[i], rotation=90, fontsize=9, ha='center')

    plt.title(f"Hypothesis Validation: 'Local Mass' per Token\n(Red bars indicate tokens triggering Global Attention)", fontsize=14)
    plt.ylabel("Local Attention Mass (0.0 - 1.0)")
    plt.ylim(0, 1.1)
    
    # Subplot 2: The Actual Attention Heatmap
    plt.subplot(2, 1, 2)
    # focus on the last 50 tokens for clarity if sequence is long
    start_idx = 0 
    sns.heatmap(
        attn_matrix[start_idx:, start_idx:], 
        xticklabels=tokens[start_idx:], 
        yticklabels=tokens[start_idx:], 
        cmap="viridis", 
        mask=np.triu(np.ones_like(attn_matrix[start_idx:, start_idx:]), k=1) # Mask upper triangle
    )
    plt.title("Attention Weights Heatmap (Layer -1)", fontsize=14)
    plt.xlabel("Key (Source)")
    plt.ylabel("Query (Target)")
    
    plt.tight_layout()
    plt.savefig("qwen_attention_experiment.png")
    print("Saved plot to 'qwen_attention_experiment.png'")
    plt.show()

if __name__ == "__main__":
    run_attention_experiment()