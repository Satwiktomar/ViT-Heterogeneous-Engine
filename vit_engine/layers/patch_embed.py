import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, Embed, Grid, Grid)
        x = self.proj(x)
        # Flatten: (B, Embed, Grid*Grid) -> Transpose: (B, N, Embed)
        x = x.flatten(2).transpose(1, 2)
        return x