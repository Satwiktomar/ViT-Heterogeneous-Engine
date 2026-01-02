from torch.utils.cpp_extension import load
import os

_cuda_lib = None

def load_cuda_kernels():
    """
    JIT Compiles and loads the C++ extension on the first call.
    """
    global _cuda_lib
    if _cuda_lib is not None:
        return _cuda_lib

    # Paths to source files
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(curr_dir, "../../")
    src_dir = os.path.join(root_dir, "csrc/cuda")
    
    sources = [
        os.path.join(src_dir, "pybind_cuda.cpp"),
        os.path.join(src_dir, "attention/flash_fwd.cu"),
        # We will add more files here as we write them
    ]
    
    # Check if files exist before trying to load (to prevent crash on empty repo)
    if not os.path.exists(sources[0]):
        print("⚠️ CUDA sources not found. Skipping compilation.")
        return None

    print("⚙️ Compiling Custom CUDA Kernels... (This may take a minute)")
    try:
        _cuda_lib = load(
            name="vit_cuda",
            sources=sources,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=True
        )
        print("✅ CUDA Kernels Loaded Successfully!")
    except Exception as e:
        print(f"❌ Failed to load CUDA kernels: {e}")
        _cuda_lib = None
        
    return _cuda_lib