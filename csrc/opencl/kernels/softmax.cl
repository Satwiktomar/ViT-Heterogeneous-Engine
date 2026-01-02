// Stable Softmax: Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
__kernel void softmax_stable(
    __global float* input,
    __global float* output,
    const int N,  // Sequence Length (columns)
    const int D   // Head Dimension (usually ignored here if input is flattened rows)
) {
    // We assume the kernel is launched with 1 thread per row
    // Global Size: (Batch * Heads * SeqLen)
    // Actually, usually Softmax is applied over the LAST dimension.
    // Let's assume input shape (Rows, Cols) and we softmax over Cols.
    
    int row_idx = get_global_id(0);
    int cols = N; 
    
    // Pointer to the start of this row
    __global float* row_in = input + row_idx * cols;
    __global float* row_out = output + row_idx * cols;

    // 1. Find Max (for numerical stability)
    float max_val = -1e20f;
    for (int i = 0; i < cols; ++i) {
        max_val = fmax(max_val, row_in[i]);
    }

    // 2. Compute Exponentials and Sum
    float sum_exp = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float val = exp(row_in[i] - max_val);
        row_out[i] = val; // Temporarily store exp value
        sum_exp += val;
    }

    // 3. Normalize
    for (int i = 0; i < cols; ++i) {
        row_out[i] /= sum_exp;
    }
}