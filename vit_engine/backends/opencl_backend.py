import os
try:
    import pyopencl as cl
except ImportError:
    cl = None

class OpenCLBackend:
    def __init__(self):
        if cl is None:
            print("⚠️ PyOpenCL not installed. OpenCL backend unavailable.")
            self.ctx = None
            return

        # 1. Select the first available GPU (or CPU if no GPU)
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found!")
            
        self.platform = platforms[0]
        self.device = self.platform.get_devices()[0]
        
        # 2. Create Context and Command Queue
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)
        print(f"✅ OpenCL Context Initialized on: {self.device.name}")

    def load_kernel(self, kernel_filename):
        """Compiles a .cl file from csrc/opencl/kernels"""
        if self.ctx is None: return None
        
        # Resolve path
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        kernel_path = os.path.join(curr_dir, "../../csrc/opencl/kernels", kernel_filename)
        
        if not os.path.exists(kernel_path):
            print(f"❌ Kernel file not found: {kernel_path}")
            return None
            
        with open(kernel_path, 'r') as f:
            kernel_src = f.read()
            
        # JIT Compile
        prg = cl.Program(self.ctx, kernel_src).build()
        return prg

# Singleton instance
_backend_instance = None

def get_opencl_ctx():
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = OpenCLBackend()
    return _backend_instance