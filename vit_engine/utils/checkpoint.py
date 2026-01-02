import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoints/best_model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(state, path)
    print(f"💾 Checkpoint saved: {path}")

def load_checkpoint(model, optimizer=None, path="checkpoints/best_model.pth", device="cuda"):
    if not os.path.exists(path):
        print("⚠️ No checkpoint found.")
        return 0
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"♻️ Loaded checkpoint from Epoch {checkpoint['epoch']}")
    return checkpoint['epoch']