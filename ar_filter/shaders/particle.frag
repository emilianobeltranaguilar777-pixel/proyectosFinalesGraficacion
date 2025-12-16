#version 150 core

// Particle fragment shader for trail effects
// OpenGL 3.2 compatible (GLSL 150)

in vec3 fragColor;
in float fragLife;

out vec4 outColor;

void main() {
    // Alpha fade based on life
    float alpha = fragLife * 0.9;

    // Soft circular particle (point sprite)
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);

    // Soft edge falloff
    float softness = 1.0 - smoothstep(0.3, 0.5, dist);
    alpha *= softness;

    outColor = vec4(fragColor, alpha);
}
