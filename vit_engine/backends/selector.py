import torch
import os

# Global flag to track which backend is active
_ACTIVE_BACKEND = "PYTORCH"  # Default to standard PyTorch

def get_backend():
    return _ACTIVE_BACKEND

def set_backend(backend_name: str):
    """
    Switch between: 'PYTORCH', 'CUDA_CUSTOM', 'OPENCL'
    """
    global _ACTIVE_BACKEND
    allowed = ["PYTORCH", "CUDA_CUSTOM", "OPENCL"]
    
    if backend_name not in allowed:
        raise ValueError(f"Backend must be one of {allowed}")
    
    if backend_name == "CUDA_CUSTOM" and not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not available. Falling back to PYTORCH.")
        _ACTIVE_BACKEND = "PYTORCH"
        return

    print(f"🔄 Switched Backend to: {backend_name}")
    _ACTIVE_BACKEND = backend_name

def use_custom_cuda():
    """Helper to check if we should use our custom kernels"""
    return _ACTIVE_BACKEND == "CUDA_CUSTOM"