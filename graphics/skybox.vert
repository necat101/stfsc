#version 450

layout(location = 0) in vec3 position;

layout(location = 0) out vec3 fragTexCoord;

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 proj;
} transform;

void main() {
    fragTexCoord = position;
    // Remove translation from view matrix to keep skybox centered on camera
    mat4 viewNoTrans = mat4(mat3(transform.view));
    vec4 pos = transform.proj * viewNoTrans * vec4(position, 1.0);
    gl_Position = pos.xyww; // Z = W ensures it's always at the far plane (depth = 1.0)
}
