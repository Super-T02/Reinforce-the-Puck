import torch

# Torch Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Environment Settings
MAX_MEMORY_SIZE = 100000
