import torch
import gc

class MemoryManager:
    @staticmethod
    def print_stats():
        if not torch.cuda.is_available():
            print("GPU not available.")
            return

        # Synchronize to get accurate reading
        torch.cuda.synchronize()
        
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"💾 [VRAM STATS] Allocated: {allocated:.2f}MB | Reserved: {reserved:.2f}MB | Peak: {max_allocated:.2f}MB")

    @staticmethod
    def cleanup():
        """Aggressive garbage collection to free VRAM"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("🧹 GPU Memory Cleaned")