#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec4 color;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    vec2 screenSize;
} pc;

void main() {
    // Convert pixel coords to NDC
    // Screen origin is top-left, Y increases downward
    vec2 ndc = (position / pc.screenSize) * 2.0 - 1.0;
    // Vulkan NDC has Y increasing downward by default, so (0,0) is top-left and (width,height) is bottom-right.
    
    gl_Position = vec4(ndc, 0.0, 1.0);
    outUV = uv;
    outColor = color;
}
