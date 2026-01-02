#version 450

layout(location = 0) in vec2 inUV;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D uiTexture;

void main() {
    vec4 texColor = texture(uiTexture, inUV);
    outColor = texColor * inColor;
    
    // Discard fully transparent pixels
    if (outColor.a < 0.01) {
        discard;
    }
}
