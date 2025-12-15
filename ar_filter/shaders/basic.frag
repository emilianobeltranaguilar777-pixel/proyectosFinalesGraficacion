#version 330 core

// Output color
out vec4 FragColor;

// Uniform for color and animation
uniform vec3 uColor;
uniform float uAlpha;
uniform float uTime;
uniform float uPulse;  // 0.0 = no pulse, 1.0 = full pulse

void main() {
    // Base color with optional pulse animation
    float pulse = 1.0 + uPulse * 0.3 * sin(uTime * 3.0);

    // Apply pulse to brightness
    vec3 finalColor = uColor * pulse;

    // Slight glow effect by brightening center
    finalColor = clamp(finalColor, 0.0, 1.0);

    FragColor = vec4(finalColor, uAlpha);
}
