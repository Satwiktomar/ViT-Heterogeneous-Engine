// Computes W += alpha * (A @ B)
__kernel void lora_update_weights(
    __global float* W,       // Main Weight Matrix [Out, In]
    __global const float* A, // LoRA A [Out, Rank]
    __global const float* B, // LoRA B [Rank, In]
    const int out_features,
    const int in_features,
    const int rank,
    const float alpha
) {
    // 2D Grid: (In_Features, Out_Features)
    int col = get_global_id(0); // in_features index
    int row = get_global_id(1); // out_features index
    
    if (row >= out_features || col >= in_features) return;
    
    // Compute dot product of Row A and Col B
    float delta = 0.0f;
    for (int r = 0; r < rank; ++r) {
        // A is [Out, Rank], B is [Rank, In]
        // A[row, r] * B[r, col]
        float a_val = A[row * rank + r];
        float b_val = B[r * in_features + col];
        delta += a_val * b_val;
    }
    
    // Update W (Atomic not needed because each thread owns one W index)
    int w_idx = row * in_features + col;
    W[w_idx] += delta * alpha;
}