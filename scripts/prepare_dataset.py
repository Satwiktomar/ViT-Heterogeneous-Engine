import torchvision
import numpy as np
import os

def prepare_cifar():
    print("📥 Downloading CIFAR-10...")
    # Download original dataset
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    
    print("⚙️ Processing images...")
    # Transpose from (H, W, C) to (C, H, W) for PyTorch format
    # Original: (50000, 32, 32, 3) -> New: (50000, 3, 32, 32)
    images = dataset.data.transpose(0, 3, 1, 2)
    labels = np.array(dataset.targets)
    
    # Create the output directory
    os.makedirs('data/processed', exist_ok=True)
    
    print("💾 Saving binary blobs to data/processed/...")
    # Save as raw contiguous bytes
    images.astype(np.uint8).tofile('data/processed/train_images.bin')
    labels.astype(np.int64).tofile('data/processed/train_labels.bin')
    
    print("✅ Done. Dataset is ready for high-speed custom loading.")

if __name__ == "__main__":
    prepare_cifar()