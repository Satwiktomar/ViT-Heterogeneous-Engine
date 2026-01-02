#version 430

// Define the workgroup size (standard 16x16 block)
layout(local_size_x = 16, local_size_y = 16) in;

// Inputs:
// binding 0: The raw attention scores (grayscale/float data)
layout(r32f, binding = 0) readonly uniform image2D attention_scores;

// Outputs:
// binding 1: The colorful image we will display
layout(rgba8, binding = 1) writeonly uniform image2D heatmap_output;

// Function to approximate the "Inferno" colormap (Black -> Orange -> Yellow)
vec3 colormap(float t) {
    // Simple ramp function
    float r = clamp(t * 2.0, 0.0, 1.0);
    float g = clamp(t * 2.0 - 0.5, 0.0, 1.0);
    float b = clamp(t * 2.0 - 1.5, 0.0, 1.0);
    return vec3(r, g, b); 
}

void main() {
    // Determine which pixel this thread is responsible for
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(attention_scores);
    
    // Bounds check
    if (pixel_coords.x >= size.x || pixel_coords.y >= size.y) return;

    // 1. Read the raw attention score
    float score = imageLoad(attention_scores, pixel_coords).r;
    
    // 2. Normalize/Amplify
    // Attention scores are usually tiny (e.g., 0.001). We multiply by 5.0 to make them visible.
    float intensity = clamp(score * 5.0, 0.0, 1.0);
    
    // 3. Convert intensity to color
    vec3 color = colormap(intensity);
    
    // 4. Write the pixel to the output image
    imageStore(heatmap_output, pixel_coords, vec4(color, 1.0));
}