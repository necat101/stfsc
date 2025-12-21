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
layout(location =8) out vec4 outPrevClipPos;
layout(location = 9) out vec3 outCameraPos;
layout(location = 10) out vec4 outNormalOffsetShadowPos;

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
    vec3 N = normalize(normalMatrix * normal);

    // Normal Offset Shadow Mapping: Push shadow sampling point slightly along normal
    // This is much more robust against acne than a constant depth bias
    // Use a larger offset for open-world scale scenes (556 Downtown)
    float shadowOffset = 0.5;  // Increased for large 200+ unit ground planes
    vec4 offsetWorldPos = vec4(worldPos.xyz + N * shadowOffset, 1.0);
    outNormalOffsetShadowPos = viewData.lightSpace * offsetWorldPos;
    
    // Safe tangent calculation - handle zero/degenerate tangent input
    vec3 rawT = normalMatrix * tangent.xyz;
    float tangentLen = length(rawT);
    
    vec3 T;
    vec3 B;
    
    if (tangentLen > 0.0001) {
        // Valid tangent - use Gram-Schmidt orthogonalization
        T = rawT / tangentLen;
        T = normalize(T - dot(T, N) * N);
        B = cross(N, T) * tangent.w;
    } else {
        // Degenerate tangent - generate arbitrary tangent from normal
        // Choose a vector not parallel to N
        vec3 helper = abs(N.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        T = normalize(cross(N, helper));
        B = cross(N, T);
    }
    
    outNormal = N;
    outTangent = T;
    outBitangent = B;
    outCameraPos = viewData.cameraPos.xyz;
}
