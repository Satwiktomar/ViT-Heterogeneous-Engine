import unittest
import torch
from vit_engine.models.vit import VisionTransformer, ViT_Tiny_Config

class TestConsistency(unittest.TestCase):
    def test_batch_invariance(self):
        """Output for image X should be same whether it's alone or in a batch"""
        model = VisionTransformer(ViT_Tiny_Config).eval()
        
        x = torch.randn(2, 3, 32, 32)
        
        # Pass full batch
        out_batch = model(x)
        
        # Pass single items
        out_1 = model(x[0:1])
        out_2 = model(x[1:2])
        
        self.assertTrue(torch.allclose(out_batch[0], out_1[0], atol=1e-5))
        self.assertTrue(torch.allclose(out_batch[1], out_2[0], atol=1e-5))
        print("✅ Batch Consistency Test Passed")

if __name__ == "__main__":
    unittest.main()