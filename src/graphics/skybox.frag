#version 450

layout(location = 0) in vec3 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 dir = normalize(fragTexCoord);
    
    // Simple gradient: Blue up, White horizon, Gray down
    float t = 0.5 * (dir.y + 1.0);
    
    vec3 topColor = vec3(0.2, 0.4, 0.8); // Sky Blue
    vec3 bottomColor = vec3(0.8, 0.8, 0.8); // Horizon White/Gray
    
    // Mix based on Y direction
    vec3 color = mix(bottomColor, topColor, max(dir.y, 0.0));
    
    // Darker ground
    if (dir.y < 0.0) {
        color = vec3(0.2);
    }
    
    outColor = vec4(color, 1.0);
}
