import torch
import os
from vit_engine.models.vit import VisionTransformer
from vit_engine.models.config import ViT_Tiny_Config

def export():
    # 1. Load Model
    model = VisionTransformer(ViT_Tiny_Config)
    model.eval()
    
    # 2. Create Dummy Input
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # 3. Define Output Path
    output_dir = "deployment"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vit_tiny.onnx")
    
    print(f"📦 Exporting model to {output_path}...")
    
    # 4. Export
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )
    print("✅ Model exported successfully.")

if __name__ == "__main__":
    export()