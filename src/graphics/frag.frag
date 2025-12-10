#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D texSampler;

void main() {
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.2)); // Hardcoded sun direction
    
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 ambient = vec3(0.1);
    
    vec4 texColor = texture(texSampler, fragUV);
    
    vec3 result = (ambient + diff) * fragColor * texColor.rgb;
    outColor = vec4(result, 1.0);
}
