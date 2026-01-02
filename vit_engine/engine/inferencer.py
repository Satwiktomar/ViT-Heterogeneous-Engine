import torch
import time
from vit_engine.backends.selector import use_custom_cuda

def benchmark_inference(model, input_shape=(1, 3, 32, 32), iterations=100, device="cuda"):
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    print("🔥 Warming up GPU...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    print(f"⚡ Benchmarking Inference (Backend: {'CUSTOM CUDA' if use_custom_cuda() else 'PYTORCH'})...")
    
    start_time = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / iterations * 1000 # in ms
    print(f"⏱️ Average Latency: {avg_latency:.2f} ms")
    return avg_latency