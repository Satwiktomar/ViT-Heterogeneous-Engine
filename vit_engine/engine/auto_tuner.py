import torch
import time
from vit_engine.backends import selector

def tune_system(model, sample_input):
    """
    Runs a quick race between PyTorch and Custom CUDA (if available)
    and selects the winner.
    """
    print("\n🏎️  Auto-Tuner: Starting Kernel Race...")
    
    # 1. Benchmark PyTorch
    selector.set_backend("PYTORCH")
    start = time.time()
    model(sample_input)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    
    # 2. Benchmark Custom CUDA (only if compiled)
    # Note: In a real scenario, we would wrap this in a try/except
    # because the kernel might not be compiled yet.
    try:
        selector.set_backend("CUDA_CUSTOM")
        # We need a dummy run to trigger JIT compilation if lazy loading
        model(sample_input) 
        start = time.time()
        model(sample_input)
        torch.cuda.synchronize()
        cuda_time = time.time() - start
    except Exception as e:
        print(f"⚠️ Custom CUDA failed: {e}")
        cuda_time = 999.0
        
    print(f"   PyTorch Time: {pytorch_time:.5f}s")
    print(f"   CUDA Time:    {cuda_time:.5f}s")
    
    if cuda_time < pytorch_time:
        print("🏆 Winner: CUSTOM CUDA")
        selector.set_backend("CUDA_CUSTOM")
    else:
        print("🏆 Winner: PYTORCH (Custom kernel might need optimization)")
        selector.set_backend("PYTORCH")