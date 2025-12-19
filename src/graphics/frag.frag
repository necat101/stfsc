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

// Dynamic Lighting - Maximum lights for mobile VR
const int MAX_LIGHTS = 256;

// GPU Light data structure (matches Rust GpuLightData)
struct LightData {
    vec4 position_type;     // xyz = position, w = type (0=point, 1=spot, 2=directional)
    vec4 direction_range;   // xyz = direction, w = range
    vec4 color_intensity;   // xyz = color, w = intensity
    vec4 cone_shadow;       // x = cos(inner), y = cos(outer), z = 1/(cos_inner-cos_outer), w = shadow_index
};

// Light Uniform Buffer
layout(set = 0, binding = 2) uniform LightUBO {
    LightData lights[MAX_LIGHTS];
    int numLights;
    vec4 ambient;           // xyz = ambient color, w = unused
} lightData;

// Constants
const float PI = 3.14159265359;

// Light types
const int LIGHT_POINT = 0;
const int LIGHT_SPOT = 1;
const int LIGHT_DIRECTIONAL = 2;

// Shadow calculation (for main directional light)
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
    float bias = 0.003;
    
    // PCF (Percentage Closer Filtering)
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

// Calculate attenuation for point/spot lights
float calculateAttenuation(float distance, float range) {
    // Smooth falloff with range limit
    float attenuation = 1.0 / (distance * distance + 1.0);
    float rangeAttenuation = clamp(1.0 - pow(distance / range, 4.0), 0.0, 1.0);
    return attenuation * rangeAttenuation * rangeAttenuation;
}

// Calculate spot light cone attenuation
float calculateSpotCone(vec3 lightDir, vec3 spotDir, float cosInner, float cosOuter, float coneRangeInv) {
    float cosAngle = dot(lightDir, -spotDir);
    return clamp((cosAngle - cosOuter) * coneRangeInv, 0.0, 1.0);
}

// Calculate contribution from a single dynamic light
vec3 calculateLightContribution(
    LightData light,
    vec3 worldPos,
    vec3 N,
    vec3 V,
    vec3 albedo,
    float metallic,
    float roughness,
    vec3 F0
) {
    int lightType = int(light.position_type.w);
    vec3 lightColor = light.color_intensity.xyz;
    float intensity = light.color_intensity.w;
    float range = light.direction_range.w;
    
    vec3 L;
    float attenuation = 1.0;
    
    if (lightType == LIGHT_DIRECTIONAL) {
        // Directional light - direction is constant
        L = normalize(-light.direction_range.xyz);
        // No distance attenuation for directional lights
    } else {
        // Point or Spot light
        vec3 lightPos = light.position_type.xyz;
        vec3 toLight = lightPos - worldPos;
        float distance = length(toLight);
        
        // Minimum distance to prevent div-by-zero
        distance = max(distance, 0.001);
        L = toLight / distance;
        
        // Distance attenuation
        attenuation = calculateAttenuation(distance, range);
        
        // Spot light cone
        if (lightType == LIGHT_SPOT) {
            vec3 spotDir = normalize(light.direction_range.xyz);
            float cosInner = light.cone_shadow.x;
            float cosOuter = light.cone_shadow.y;
            float coneRangeInv = light.cone_shadow.z;
            attenuation *= calculateSpotCone(L, spotDir, cosInner, cosOuter, coneRangeInv);
        }
    }
    
    // Early out if light contribution is negligible
    if (attenuation < 0.001) {
        return vec3(0.0);
    }
    
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    
    // Skip if facing away
    if (NdotL <= 0.0) {
        return vec3(0.0);
    }
    
    // BRDF calculation
    float NDF = DistributionGGX(N, H, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
    
    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;
    
    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
    vec3 specular = numerator / denominator;
    
    // Final radiance from this light
    vec3 radiance = lightColor * intensity * attenuation;
    
    return (kD * albedo / PI + specular) * radiance * NdotL;
}

void main() {
    // Sample textures
    // If vertex color is white (1,1,1), use texture as-is (default)
    // If vertex color is non-white, use it as a tint multiplier
    vec4 texColor = texture(albedoMap, inUV);
    vec3 albedo = texColor.rgb;
    
    // Check if color is "neutral" (white) - if so, use pure texture
    // Otherwise, tint the texture with the vertex color
    float colorBrightness = (inColor.r + inColor.g + inColor.b) / 3.0;
    if (colorBrightness < 0.99) {
        // Non-white color means "tint mode" - multiply texture by color
        albedo = texColor.rgb * inColor;
    }
    vec3 normalSample = texture(normalMap, inUV).rgb;
    normalSample = normalize(normalSample * 2.0 - 1.0);
    
    // Safe normalization for TBN matrix - handle degenerate tangent/bitangent
    // This prevents NaN when tangent vectors are zero or near-zero
    vec3 N_world = normalize(inNormal);
    vec3 T = inTangent;
    vec3 B = inBitangent;
    
    float tangentLen = length(T);
    float bitangentLen = length(B);
    
    // Check if tangent/bitangent are valid (non-zero length)
    bool validTBN = tangentLen > 0.0001 && bitangentLen > 0.0001;
    
    vec3 N;
    if (validTBN) {
        // Normalize and create TBN matrix
        T = T / tangentLen;
        B = B / bitangentLen;
        mat3 TBN = mat3(T, B, N_world);
        N = normalize(TBN * normalSample);
        
        // Final safety check - if N became NaN, fall back to world normal
        if (any(isnan(N)) || any(isinf(N))) {
            N = N_world;
        }
    } else {
        // Fall back to world normal when TBN is degenerate
        N = N_world;
    }
    
    vec3 mr = texture(metallicRoughnessMap, inUV).rgb;
    float metallic = mr.r;
    float roughness = max(mr.g, 0.04); // Prevent zero roughness
    
    // View direction with safe normalization
    vec3 viewDir = inCameraPos - inWorldPos;
    float viewDist = length(viewDir);
    vec3 V = viewDist > 0.0001 ? viewDir / viewDist : vec3(0.0, 1.0, 0.0);
    
    // F0 (Fresnel at normal incidence)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);
    
    // Accumulate lighting from all dynamic lights
    vec3 Lo = vec3(0.0);
    
    // Process dynamic lights
    int numLights = min(lightData.numLights, MAX_LIGHTS);
    for (int i = 0; i < numLights; i++) {
        Lo += calculateLightContribution(
            lightData.lights[i],
            inWorldPos,
            N, V,
            albedo, metallic, roughness, F0
        );
    }
    
    // Fallback: If no dynamic lights, use hardcoded directional light
    // This maintains backwards compatibility with existing scenes
    if (numLights == 0) {
        vec3 L = normalize(vec3(20.0, 50.0, 20.0));
        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        
        float NDF = DistributionGGX(N, H, roughness);
        float G = GeometrySmith(N, V, L, roughness);
        vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;
        
        vec3 numerator = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * NdotL + 0.0001;
        vec3 specular = numerator / denominator;
        
        // Shadow (only for fallback directional light)
        float shadow = ShadowCalculation(inLightSpacePos);
        
        Lo = (kD * albedo / PI + specular) * vec3(3.0) * NdotL * (1.0 - shadow);
    } else {
        // For dynamic lights, apply shadow to first directional light if present
        if (lightData.lights[0].position_type.w == float(LIGHT_DIRECTIONAL)) {
            float shadow = ShadowCalculation(inLightSpacePos);
            // Reduce first light's contribution by shadow amount
            // (simplified - proper implementation would track per-light shadows)
            Lo *= mix(vec3(1.0), vec3(1.0 - shadow * 0.5), vec3(1.0));
        }
    }
    
    // Ambient lighting
    vec3 ambient = lightData.ambient.xyz * albedo;
    if (numLights == 0) {
        ambient = vec3(0.15) * albedo; // Fallback ambient
    }
    
    vec3 color = ambient + Lo;
    
    // Tone mapping (Reinhard)
    color = color / (color + vec3(1.0));
    
    // Gamma correction
    color = pow(color, vec3(1.0/2.2));
    
    outColor = vec4(color, 1.0);
    
    // Motion Vector Calculation for AppSW
    vec3 ndc = inClipPos.xyz / inClipPos.w;
    vec3 prevNdc = inPrevClipPos.xyz / inPrevClipPos.w;
    
    vec2 motion = (ndc.xy - prevNdc.xy);
    // XR_FB_space_warp spec: "The values... are in NDC space [-1, 1]."
    
    outMotionVector = vec4(motion, 0.0, 1.0);
}
