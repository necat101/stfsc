#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 color;
layout(location = 4) in vec4 tangent;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outUV;
layout(location = 3) out vec3 outColor;
layout(location = 4) out vec3 outTangent;
layout(location = 5) out vec3 outBitangent;
layout(location = 6) out vec4 outLightSpacePos;
layout(location = 7) out vec4 outClipPos;
layout(location = 8) out vec4 outPrevClipPos;
layout(location = 9) out vec3 outCameraPos;

struct InstanceData {
    mat4 model;
    mat4 prevModel;
    vec4 color;
};

// Set 0 Binding 1: Instance Buffer
layout(std430, set = 0, binding = 1) readonly buffer InstanceBuffer {
    InstanceData instances[];
} instanceData;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    mat4 prevViewProj;
    mat4 lightSpace;
    vec4 cameraPos; // xyz = camera position, w unused
} viewData;

void main() {
    mat4 model = instanceData.instances[gl_InstanceIndex].model;
    mat4 prevModel = instanceData.instances[gl_InstanceIndex].prevModel;
    vec4 instanceColor = instanceData.instances[gl_InstanceIndex].color;

    vec4 worldPos = model * vec4(position, 1.0);
    gl_Position = viewData.viewProj * worldPos;
    
    outClipPos = gl_Position;
    
    vec4 prevWorldPos = prevModel * vec4(position, 1.0);
    outPrevClipPos = viewData.prevViewProj * prevWorldPos;

    outWorldPos = worldPos.xyz;
    outUV = uv;
    outColor = color * instanceColor.rgb; // Multiply vertex color with instance color
    outLightSpacePos = viewData.lightSpace * worldPos;
    
    // Normal Matrix
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    vec3 T = normalize(normalMatrix * tangent.xyz);
    vec3 N = normalize(normalMatrix * normal);
    T = normalize(T - dot(T, N) * N); // Gram-Schmidt
    vec3 B = cross(N, T) * tangent.w;
    
    outNormal = N;
    outTangent = T;
    outBitangent = B;
    outCameraPos = viewData.cameraPos.xyz;
}
