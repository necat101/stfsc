#version 450
layout(location = 0) in vec3 position;
// Other attributes ignored

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 proj;
} transform;

void main() {
    // Standard MVP transform
    gl_Position = transform.proj * transform.view * transform.model * vec4(position, 1.0);
}
