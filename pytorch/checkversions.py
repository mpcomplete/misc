import sys, os
import torch

print("Python exe:", sys.executable)
print("Torch version:", torch.__version__)
print("Torch path:", torch.__file__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))