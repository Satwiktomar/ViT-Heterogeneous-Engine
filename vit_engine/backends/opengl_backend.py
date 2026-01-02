try:
    import moderngl
    import numpy as np
except ImportError:
    moderngl = None

class OpenGLBackend:
    def __init__(self, width=1024, height=768, headless=True):
        if moderngl is None:
            print("⚠️ ModernGL not installed. Visualization unavailable.")
            self.ctx = None
            return
            
        # If headless (server), we use standalone context
        try:
            self.ctx = moderngl.create_context(standalone=True)
            print("✅ OpenGL Context (Headless) Initialized")
        except Exception as e:
            print(f"❌ Failed to init OpenGL: {e}")
            self.ctx = None

    def create_texture_from_tensor(self, tensor):
        """
        Converts a PyTorch tensor (GPU) directly to an OpenGL Texture
        Note: This usually requires CUDA-GL Interop (advanced),
        for now we assume CPU sync for simplicity in Phase 1.
        """
        if self.ctx is None: return None
        
        data = tensor.detach().cpu().numpy().astype('f4')
        texture = self.ctx.texture(
            (tensor.shape[1], tensor.shape[0]), # W, H
            1, # Components
            data.tobytes()
        )
        return texture

# Singleton
_gl_instance = None

def get_opengl_ctx():
    global _gl_instance
    if _gl_instance is None:
        _gl_instance = OpenGLBackend()
    return _gl_instance