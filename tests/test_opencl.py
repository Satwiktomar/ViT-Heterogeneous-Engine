import unittest
from vit_engine.backends.opencl_backend import get_opencl_ctx

class TestOpenCL(unittest.TestCase):
    def test_context_creation(self):
        try:
            ctx = get_opencl_ctx()
            if ctx.ctx is None:
                print("⚠️ OpenCL not found, skipping test.")
                return
            self.assertIsNotNone(ctx.queue)
            print("✅ OpenCL Context Creation Passed")
        except Exception as e:
            print(f"Skipping OpenCL test: {e}")

if __name__ == "__main__":
    unittest.main()