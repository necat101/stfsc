#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inColor;
layout(location = 4) in vec3 inTangent;
layout(location = 5) in vec3 inBitangent;
layout(location = 6) in vec4 inLightSpacePos;
layout(location = 7) in vec4 inClipPos;
layout(location = 8) in vec4 inPrevClipPos;
layout(location = 9) in vec3 inCameraPos;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMotionVector;

layout(set = 1, binding = 0) uniform sampler2D albedoMap;
layout(set = 1, binding = 1) uniform sampler2D normalMap;
layout(set = 1, binding = 2) uniform sampler2D metallicRoughnessMap;
layout(set = 0, binding = 0) uniform sampler2D shadowMap;

// Constants (Keep previous constants)
const float PI = 3.14159265359;
const vec3 lightPos = vec3(10.0, 10.0, 10.0); // Not used if using Directional
const vec3 camPos = vec3(0.0, 0.0, 0.0); 

// ... GGX functions ... Note: I am not replacing them, so start line should be after imports.

float ShadowCalculation(vec4 fragPosLightSpace) {
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;
    
    // Check if fragment is outside shadow map frustum - return fully lit
    if(projCoords.z > 1.0 || projCoords.z < 0.0)
        return 0.0;
    if(projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0)
        return 0.0;
        
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    float currentDepth = projCoords.z;
    
    vec3 normal = normalize(inNormal);
    vec3 lightDir = normalize(vec3(20.0, 50.0, 20.0)); // Match shadow light offset
    // Reduced bias to prevent peter-panning (objects appearing to float)
    float bias = 0.003;
    
    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
        }    
    }
    shadow /= 9.0;
    
    return shadow;
}

// PBR Functions
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    // ... previous setup ...
    vec3 albedo = texture(albedoMap, inUV).rgb * inColor;
    vec3 normalSample = texture(normalMap, inUV).rgb;
    normalSample = normalize(normalSample * 2.0 - 1.0);
    
    mat3 TBN = mat3(normalize(inTangent), normalize(inBitangent), normalize(inNormal));
    vec3 N = normalize(TBN * normalSample);
    
    vec3 mr = texture(metallicRoughnessMap, inUV).rgb;
    float metallic = mr.r;
    float roughness = mr.g;

    // Use actual camera position for correct view vector
    vec3 V = normalize(inCameraPos - inWorldPos);
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    vec3 L = normalize(vec3(20.0, 50.0, 20.0)); // Match shadow light offset 
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);

    // BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    vec3 specular = numerator / denominator;

    // Shadow
    float shadow = ShadowCalculation(inLightSpacePos);
    
    vec3 Lo = (kD * albedo / PI + specular) * vec3(3.0) * NdotL * (1.0 - shadow); 

    vec3 ambient = vec3(0.15) * albedo; // Higher ambient to avoid complete darkness
    vec3 color = ambient + Lo;

    // Gamma correction
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));

    outColor = vec4(color, 1.0);
    
    // Motion Vector Calculation for AppSW
    vec3 ndc = inClipPos.xyz / inClipPos.w;
    vec3 prevNdc = inPrevClipPos.xyz / inPrevClipPos.w;
    
    vec2 motion = (ndc.xy - prevNdc.xy); // check if scale needed? openxr spec usually wants normalized [-1..1] delta? or UV delta?
    // XR_FB_space_warp spec: "The values... are in NDC space [-1, 1]."
    // So delta in NDC is correct.
    
    // Note: Some engines subtract Jitter. We don't have TAA jitter here.
    
    outMotionVector = vec4(motion, 0.0, 1.0); // xy = motion, zw = unused (alpha 1.0 just in case)
}
