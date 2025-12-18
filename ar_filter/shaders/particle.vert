#version 150 core

// Particle vertex shader for trail effects
// OpenGL 3.2 compatible (GLSL 150)

in vec3 position;
in vec3 color;
in float life;

out vec3 fragColor;
out float fragLife;

uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * vec4(position, 1.0);

    // Point size based on life (larger when fresh, smaller when fading)
    gl_PointSize = 6.0 * life;

    fragColor = color;
    fragLife = life;
}
