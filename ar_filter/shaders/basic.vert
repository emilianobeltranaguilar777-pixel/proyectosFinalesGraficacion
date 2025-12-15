#version 330 core

// Input vertex position
layout(location = 0) in vec2 aPos;

// Uniform for transformation
uniform mat4 uProjection;
uniform vec2 uOffset;
uniform float uScale;

void main() {
    // Apply scale and offset
    vec2 pos = aPos * uScale + uOffset;

    // Output position with projection
    gl_Position = uProjection * vec4(pos, 0.0, 1.0);
}
