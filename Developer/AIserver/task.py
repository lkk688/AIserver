import random
import numpy as np
import torch


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Set seed for reproducibility
set_seed(42)

# Your task code here
print("Task executed with seed 42")
