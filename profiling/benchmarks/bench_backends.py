import time
import torch
from vit_engine.models.vit import VisionTransformer, ViT_Tiny_Config
from vit_engine.backends import selector

def benchmark_all():
    model = VisionTransformer(ViT_Tiny_Config).cuda()
    input_data = torch.randn(16, 3, 32, 32).cuda()
    
    backends = ["PYTORCH"]
    # Check if we can add others
    try:
        selector.set_backend("CUDA_CUSTOM")
        backends.append("CUDA_CUSTOM")
    except: pass
    
    results = {}
    
    for backend in backends:
        selector.set_backend(backend)
        
        # Warmup
        model(input_data)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(50):
            model(input_data)
        torch.cuda.synchronize()
        
        results[backend] = (time.time() - start) / 50 * 1000
        print(f"Backend [{backend}]: {results[backend]:.3f} ms")

if __name__ == "__main__":
    benchmark_all()