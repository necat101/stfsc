#version 450

layout (location = 0) out vec2 outUV;

void main() 
{
    // Generates a full-screen triangle from vertex index
    // Correct UVs: (0,0), (2,0), (0,2) covers the clip space (-1,-1) to (1,1)
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0f - 1.0f, 0.0f, 1.0f);
}
