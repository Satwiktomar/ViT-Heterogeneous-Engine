from dataclasses import dataclass

@dataclass
class ViTConfig:
    # Model Architecture
    img_size: int = 32
    patch_size: int = 4
    in_chans: int = 3
    num_classes: int = 10
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    
    # Dropout / Regularization
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    
    # Training
    batch_size: int = 128
    lr: float = 3e-4
    epochs: int = 50
    weight_decay: float = 0.01

# Preset Configurations
ViT_Tiny_Config = ViTConfig(embed_dim=192, depth=12, num_heads=3)
ViT_Small_Config = ViTConfig(embed_dim=384, depth=12, num_heads=6)