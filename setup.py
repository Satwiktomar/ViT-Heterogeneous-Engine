import os
# Force PyTorch to accept the Arch Linux CUDA 13.0 compiler
os.environ["TORCH_DONT_CHECK_COMPILER_VERSION"] = "1"

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Helper to find files
sources = [
    "csrc/cuda/pybind_cuda.cpp",
    "csrc/cuda/attention/flash_fwd.cu",
    # "csrc/cuda/attention/flash_bwd.cu", # Uncomment when implemented
]

setup(
    name='vit_cuda_driver',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name='vit_engine.vit_cuda', # This puts the .so inside vit_engine folder
            sources=sources,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)