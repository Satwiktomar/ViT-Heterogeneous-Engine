import torch
import torch.nn as nn
from .config import ViTConfig
from ..layers.patch_embed import PatchEmbed
from ..layers.attention import Attention

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_classes = config.num_classes
        self.embed_dim = config.embed_dim

        self.patch_embed = PatchEmbed(
            img_size=config.img_size, patch_size=config.patch_size, 
            in_chans=config.in_chans, embed_dim=config.embed_dim
        )
        
        num_patches = self.patch_embed.num_patches

        # Learnable Position Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.drop_rate)

        # Transformer Encoder
        self.blocks = nn.ModuleList([
            Block(
                dim=config.embed_dim, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, 
                qkv_bias=config.qkv_bias, drop=config.drop_rate, attn_drop=config.attn_drop_rate
            )
            for _ in range(config.depth)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add Positional Embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer Blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        
        # Classifier (only use CLS token)
        return self.head(x[:, 0])