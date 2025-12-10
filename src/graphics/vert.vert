#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 color;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 proj;
} transform;

void main() {
    gl_Position = transform.proj * transform.view * transform.model * vec4(position, 1.0);
    fragColor = color;
    // Transform normal to world space (assuming uniform scaling for now)
    fragNormal = mat3(transform.model) * normal;
    fragUV = uv;
}
