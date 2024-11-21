import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.enabled)
print("CUDA available:", torch.cuda.is_available())
