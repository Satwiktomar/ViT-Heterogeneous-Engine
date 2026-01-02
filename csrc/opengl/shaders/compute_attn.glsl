#version 430
// Define the workgroup size (e.g., 16x16 threads)
layout(local_size_x = 16, local_size_y = 16) in;

// Use Shader Storage Buffer Objects (SSBOs) for large data
// Binding numbers must match the host code
layout(std430, binding = 0) readonly buffer QBuffer {
    float Q[];
};
layout(std430, binding = 1) readonly buffer KBuffer {
    float K[];
};
// Output: The N x N attention matrix
layout(std430, binding = 2) writeonly buffer ScoreBuffer {
    float Scores[];
};

// Uniforms passed from host
uniform int seq_len;    // N
uniform int head_dim;   // D
uniform float scale;    // 1 / sqrt(D)

void main() {
    // Global invocation IDs define which part of the output matrix we compute
    uint row = gl_GlobalInvocationID.y; // Query Index (0 to N-1)
    uint col = gl_GlobalInvocationID.x; // Key Index (0 to N-1)

    if (row >= seq_len || col >= seq_len) return;

    // Calculate Dot Product: Q[row] . K[col]^T
    float dot_prod = 0.0;
    for (int d = 0; d < head_dim; ++d) {
        // Assuming flattened [N, D] layout
        uint q_idx = row * head_dim + d;
        uint k_idx = col * head_dim + d;
        dot_prod += Q[q_idx] * K[k_idx];
    }

    // Scale the result
    // Note: Softmax is usually done in a subsequent pass or combined here
    // For this example, we just output the raw scaled score.
    uint score_idx = row * seq_len + col;
    Scores[score_idx] = dot_prod * scale;
}