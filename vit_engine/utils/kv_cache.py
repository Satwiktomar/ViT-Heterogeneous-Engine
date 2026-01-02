import torch

class KVCache:
    """
    Manages Key-Value Cache for Auto-Regressive Inference.
    """
    def __init__(self, max_batch_size, max_seq_len, n_heads, head_dim, device="cuda"):
        self.max_seq_len = max_seq_len
        self.k_cache = torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), device=device, dtype=torch.float16)
        self.v_cache = torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), device=device, dtype=torch.float16)
        self.curr_pos = 0

    def update(self, k, v, pos_start):
        """
        k, v: (Batch, Heads, SeqLen_New, HeadDim)
        """
        batch, _, seq_len, _ = k.shape
        # Store the new keys/values in the pre-allocated buffer
        self.k_cache[:batch, :, pos_start:pos_start+seq_len, :] = k
        self.v_cache[:batch, :, pos_start:pos_start+seq_len, :] = v
        
    def get_current(self, batch_size, pos_end):
        """Retreive all past keys/values up to current position"""
        return (
            self.k_cache[:batch_size, :, :pos_end, :],
            self.v_cache[:batch_size, :, :pos_end, :]
        )

    def reset(self):
        self.curr_pos = 0
        self.k_cache.zero_()
        self.v_cache.zero_()