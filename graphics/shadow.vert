#version 450
layout(location = 0) in vec3 position;
// Other attributes ignored

struct InstanceData {
    mat4 model;
    mat4 prevModel;
};

layout(std430, set = 0, binding = 1) readonly buffer InstanceBuffer {
    InstanceData instances[];
} instanceData;

layout(push_constant) uniform PushConstants {
    mat4 viewProj; // Light View * Light Proj
} transform;

void main() {
    mat4 model = instanceData.instances[gl_InstanceIndex].model;
    gl_Position = transform.viewProj * model * vec4(position, 1.0);
}
