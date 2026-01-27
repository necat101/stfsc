#version 450

// Skinned Vertex Shader for STFSC Engine
// GPU-accelerated skeletal animation with up to 4 bone influences per vertex
// Optimized for Quest 3's 2.4 TFLOPS - 556 Downtown animated NPC rendering

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 color;
layout(location = 4) in vec4 tangent;
layout(location = 5) in uvec4 boneIndices;  // Up to 4 bone influences
layout(location = 6) in vec4 boneWeights;   // Normalized weights (sum to 1.0)

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
layout(location = 10) out vec4 outNormalOffsetShadowPos;

struct InstanceData {
    mat4 model;
    mat4 prevModel;
    vec4 color;
    float metallic;
    float roughness;
    vec2 _padding;
};

// Set 0 Binding 1: Instance Buffer
layout(std430, set = 0, binding = 1) readonly buffer InstanceBuffer {
    InstanceData instances[];
} instanceData;

// Set 0 Binding 3: Bone Matrices (max 128 bones per skeleton)
layout(std140, set = 0, binding = 3) uniform BoneMatrices {
    mat4 bones[128];
} boneData;

layout(set = 0, binding = 4) uniform GlobalUBO {
    mat4 viewProj;
    mat4 prevViewProj;
    mat4 lightSpace;
    vec4 cameraPos; // xyz = camera position, w = ambient
    vec4 lightDir;  // xyz = light direction, w unused
} globalData;

// Compute skinning matrix from bone influences
mat4 computeSkinMatrix() {
    mat4 skinMatrix = 
        boneData.bones[boneIndices.x] * boneWeights.x +
        boneData.bones[boneIndices.y] * boneWeights.y +
        boneData.bones[boneIndices.z] * boneWeights.z +
        boneData.bones[boneIndices.w] * boneWeights.w;
    return skinMatrix;
}

// Apply skinning to a position
vec4 skinPosition(vec3 pos, mat4 skinMatrix) {
    return skinMatrix * vec4(pos, 1.0);
}

// Apply skinning to a normal (using the 3x3 part of the skin matrix)
vec3 skinNormal(vec3 n, mat4 skinMatrix) {
    return normalize(mat3(skinMatrix) * n);
}

// Apply skinning to a tangent
vec3 skinTangent(vec3 t, mat4 skinMatrix) {
    return normalize(mat3(skinMatrix) * t);
}

void main() {
    mat4 model = instanceData.instances[gl_InstanceIndex].model;
    mat4 prevModel = instanceData.instances[gl_InstanceIndex].prevModel;
    vec4 instanceColor = instanceData.instances[gl_InstanceIndex].color;

    // Compute skin matrix from bone influences
    mat4 skinMatrix = computeSkinMatrix();

    // Apply skinning to position
    vec4 skinnedPos = skinPosition(position, skinMatrix);
    vec4 worldPos = model * skinnedPos;
    gl_Position = globalData.viewProj * worldPos;
    // gl_Position = vec4(position.x * 0.0001, position.y * 0.0001, 0.5, 1.0); // DEBUG: SQUASH SKINNED TO CENTER
    
    outClipPos = gl_Position;
    
    // Previous frame skinned position (for motion vectors)
    // Note: For fully accurate motion blur, you'd need previous frame bone matrices
    // This approximation uses current skinning with previous model matrix
    vec4 prevWorldPos = prevModel * skinnedPos;
    outPrevClipPos = globalData.prevViewProj * prevWorldPos;

    outWorldPos = worldPos.xyz;
    outUV = uv;
    outColor = color * instanceColor.rgb;
    outLightSpacePos = globalData.lightSpace * worldPos;

    // Skin the normal using the combined skin+model normal matrix
    mat4 fullTransform = model * skinMatrix;
    mat3 normalMatrix = transpose(inverse(mat3(fullTransform)));
    vec3 N = normalize(normalMatrix * normal);

    // Normal Offset Shadow Mapping
    float shadowOffset = 0.5;
    vec4 offsetWorldPos = vec4(worldPos.xyz + N * shadowOffset, 1.0);
    outNormalOffsetShadowPos = globalData.lightSpace * offsetWorldPos;
    // outNormalOffsetShadowPos = vec4(0.0); // DEBUG: Force Zero to prevent NaNs
    
    // Skin and transform tangent
    vec3 rawT = normalMatrix * tangent.xyz;
    float tangentLen = length(rawT);
    
    vec3 T;
    vec3 B;
    
    if (tangentLen > 0.0001) {
        T = rawT / tangentLen;
        T = normalize(T - dot(T, N) * N);
        B = cross(N, T) * tangent.w;
    } else {
        vec3 helper = abs(N.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        T = normalize(cross(N, helper));
        B = cross(N, T);
    }
    
    outNormal = N;
    outTangent = T;
    outBitangent = B;
    outCameraPos = globalData.cameraPos.xyz;
}
