#version 450

layout (binding = 0) uniform sampler2D img;
layout (location = 0) in vec2 inUV;
layout (location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    float sharpening;
} push;

void main() {
    // Contrast Adaptive Sharpening (approximate)
    vec2 size = textureSize(img, 0);
    vec2 off = 1.0 / size;
    
    vec3 c11 = texture(img, inUV).rgb;
    vec3 c01 = texture(img, inUV + vec2(-off.x, 0)).rgb;
    vec3 c21 = texture(img, inUV + vec2(off.x, 0)).rgb;
    vec3 c10 = texture(img, inUV + vec2(0, -off.y)).rgb;
    vec3 c12 = texture(img, inUV + vec2(0, off.y)).rgb;
    
    // Luma = 0.299R + 0.587G + 0.114B
    vec3 luma_coeffs = vec3(0.299, 0.587, 0.114);
    float l11 = dot(c11, luma_coeffs);
    float l01 = dot(c01, luma_coeffs);
    float l21 = dot(c21, luma_coeffs);
    float l10 = dot(c10, luma_coeffs);
    float l12 = dot(c12, luma_coeffs);
    
    float min_l = min(l11, min(min(l01, l21), min(l10, l12)));
    float max_l = max(l11, max(max(l01, l21), max(l10, l12)));
    
    float contrast = max_l - min_l;
    // weight formula: -0.125 * amount * (1.0 - normalized_contrast)
    float weight = -0.125 * push.sharpening * (1.0 - (contrast / max(max_l, 0.001)));
    
    vec3 sharpened = c11 + (c01 + c21 + c10 + c12 - 4.0 * c11) * weight;
    outColor = vec4(clamp(sharpened, 0.0, 1.0), 1.0);
}
