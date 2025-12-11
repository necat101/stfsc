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

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 viewProj;
    mat4 prevViewProj;
    mat4 lightSpace;
} transform;

void main() {
    vec4 worldPos = transform.model * vec4(position, 1.0);
    gl_Position = transform.viewProj * worldPos;
    
    outClipPos = gl_Position;
    outPrevClipPos = transform.prevViewProj * worldPos; // Assuming static object for now

    outWorldPos = worldPos.xyz;
    outUV = uv;
    outColor = color;
    outLightSpacePos = transform.lightSpace * worldPos;
    
    // Normal Matrix
    mat3 normalMatrix = transpose(inverse(mat3(transform.model)));
    vec3 T = normalize(normalMatrix * tangent.xyz);
    vec3 N = normalize(normalMatrix * normal);
    T = normalize(T - dot(T, N) * N); // Gram-Schmidt
    vec3 B = cross(N, T) * tangent.w;
    
    outNormal = N;
    outTangent = T;
    outBitangent = B;
}
