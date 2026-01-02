__kernel void attention_naive(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* Output,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    // Indices: Batch*Head is implicit in global_id(1), Seq in global_id(0)
    int row = get_global_id(0); // Sequence position (0 to N-1)
    int head_idx = get_global_id(1); // Batch * Num_Heads
    
    if (row >= seq_len) return;

    int offset = head_idx * seq_len * head_dim;
    
    // 1. Calculate Scores (Row of Q vs All K)
    float max_val = -1e20f;
    // We can't malloc in OpenCL, so we re-calculate or use local mem.
    // For "Naive", we will do a 2-pass approach or just compute sum directly.
    
    // Let's implement the "Softmax inside" loop for memory efficiency
    float sum_exp = 0.0f;
    
    // Pointers to this head's data
    __global const float* q_vec = Q + offset + row * head_dim;
    
    // Need a temporary buffer for scores if we don't fuse. 
    // To keep this simple and working:
    // This kernel just computes O = Softmax(QK^T)V sequentially
    
    // Create Output Vector
    for (int d = 0; d < head_dim; ++d) {
        float res = 0.0f;
        float normalization = 0.0f;
        
        // This is O(N^2) per thread - slow but functional for fallback
        for (int col = 0; col < seq_len; ++col) {
            // Dot Product Q[row] . K[col]
            float score = 0.0f;
            for (int k = 0; k < head_dim; ++k) {
                score += q_vec[k] * K[offset + col * head_dim + k];
            }
            score *= scale;
            
            float weight = exp(score); // Unstable naive exp
            normalization += weight;
            res += weight * V[offset + col * head_dim + d];
        }
        Output[offset + row * head_dim + d] = res / (normalization + 1e-6f);
    }
}