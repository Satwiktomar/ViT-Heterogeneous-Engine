#version 330 core

// Input from Vertex Shader (texture coordinates)
in vec2 TexCoords;

// Output color to the screen
out vec4 FragColor;

// Uniforms
uniform sampler2D uiTexture; // The UI element sprite/texture
uniform vec4 uiColor;        // A tint color (e.g., for highlighting)

void main() {
    // Sample the texture color
    vec4 texColor = texture(uiTexture, TexCoords);
    
    // Apply the tint color
    // Multiplying allows tinting while keeping texture alpha
    FragColor = texColor * uiColor;
    
    // Optional: Discard transparent pixels to avoid depth issues in simple renderers
    if (FragColor.a < 0.1)
        discard;
}