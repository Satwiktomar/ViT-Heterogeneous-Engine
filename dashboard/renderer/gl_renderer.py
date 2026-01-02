import numpy as np
import torch

class AttentionVisualizer:
    def __init__(self, height=256, width=256):
        self.height = height
        self.width = width
        # In a real app, this would initialize the OpenGL context
        # For this dashboard prototype, we simulate the output

    def render_heatmap(self, attention_tensor):
        """
        Simulates the OpenGL shader output for the dashboard.
        Input: Tensor (Heads, N, N)
        Output: Numpy Array (Image)
        """
        # Take the first head and average across queries
        if attention_tensor is None:
            return np.zeros((self.height, self.width, 3))
            
        attn = attention_tensor[0].mean(dim=0).cpu().numpy()
        
        # Simple color mapping simulation (Blue -> Red)
        # Real OpenGL would do this on GPU
        norm_attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)
        
        # Create heatmap image (N x N)
        heatmap = np.stack([norm_attn, np.zeros_like(norm_attn), 1.0 - norm_attn], axis=2)
        
        # Upscale for display
        return heatmap