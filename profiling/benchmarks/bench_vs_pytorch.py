import torch
import time
from vit_engine.layers.attention import Attention
from vit_engine.backends.cuda_backend import load_cuda_kernels

def run_benchmark():
    B, N, C = 32, 64, 192
    H = 3
    x = torch.randn(B, N, C, device='cuda')
    
    print(f"🏎️ Benchmarking Attention: Batch={B}, Seq={N}, Dim={C}")
    
    # 1. PyTorch Standard
    layer_pt = Attention(dim=C, num_heads=H).cuda()
    
    # Warmup
    for _ in range(10): layer_pt(x)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        layer_pt(x)
    torch.cuda.synchronize()
    pt_time = (time.time() - start) / 100 * 1000
    
    print(f"   PyTorch Avg Time: {pt_time:.3f} ms")

    # 2. Custom CUDA (If compiled)
    cuda_ops = load_cuda_kernels()
    if cuda_ops:
        # Note: In a real benchmark, you'd call the raw kernel function here
        # similar to how we tested in test_cuda.py
        pass
        
if __name__ == "__main__":
    if torch.cuda.is_available():
        run_benchmark()
    else:
        print("Skipping benchmark (No GPU)")