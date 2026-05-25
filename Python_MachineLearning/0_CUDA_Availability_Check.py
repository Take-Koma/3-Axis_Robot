import torch
print(f"CUDA利用可能か: {torch.cuda.is_available()}")
print(f"使用デバイス名: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
