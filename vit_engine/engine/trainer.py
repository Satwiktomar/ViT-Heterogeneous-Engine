import torch
import torch.nn as nn
import torch.optim as optim
import time
from vit_engine.models.config import ViT_Tiny_Config
from vit_engine.models.vit import VisionTransformer
from vit_engine.utils.data_loader import get_dataloaders

def train():
    # 1. Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on {device}")
    
    config = ViT_Tiny_Config
    model = VisionTransformer(config).to(device)
    
    trainloader, testloader = get_dataloaders(batch_size=config.batch_size, img_size=config.img_size)
    
    # 2. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    print(f"Start Training: {config.epochs} Epochs")
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{config.epochs}] "
              f"Loss: {running_loss/len(trainloader):.4f} "
              f"Time: {epoch_time:.2f}s "
              f"Throughput: {len(trainloader.dataset)/epoch_time:.0f} img/s")

if __name__ == "__main__":
    train()