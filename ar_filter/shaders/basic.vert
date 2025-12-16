#version 150 core

// Simple vertex shader for AR filter primitives
// OpenGL 3.2 compatible (GLSL 150)

in vec3 position;
in vec3 normal;

out vec3 fragNormal;
out vec3 fragPosition;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vec4 worldPos = model * vec4(position, 1.0);
    fragPosition = worldPos.xyz;
    fragNormal = mat3(model) * normal;

    gl_Position = projection * view * worldPos;
}
