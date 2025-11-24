import torch

ckpt = torch.load("outputs/best_model.pth", map_location="cpu")

print("Typ:", type(ckpt))
if isinstance(ckpt, dict):
    print("Klucze:", ckpt.keys())
else:
    print("To nie dict")
