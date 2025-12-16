#version 150 core

// Simple fragment shader for AR filter primitives
// OpenGL 3.2 compatible (GLSL 150)

in vec3 fragNormal;
in vec3 fragPosition;

out vec4 outColor;

uniform vec3 objectColor;
uniform vec3 lightDir;
uniform float ambient;

void main() {
    // Normalize the interpolated normal
    vec3 norm = normalize(fragNormal);

    // Simple directional light
    float diff = max(dot(norm, normalize(lightDir)), 0.0);

    // Combine ambient and diffuse
    vec3 result = (ambient + diff * (1.0 - ambient)) * objectColor;

    outColor = vec4(result, 1.0);
}
