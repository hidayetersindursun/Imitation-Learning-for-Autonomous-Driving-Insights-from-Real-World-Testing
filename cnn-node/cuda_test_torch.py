import torch

if torch.cuda.is_available():
    print("CUDA is available. GPU(s) detected.")
    print("Number of GPUs available:", torch.cuda.device_count())
    print("GPU device name(s):", torch.cuda.get_device_name(0))  # Assuming there is at least one GPU
else:
    print("CUDA is not available. Only CPU is detected.")
