"""
OpenGL + GLFW application for AR filter rendering.

This is the ONLY file that uses OpenGL.
Renders:
- Camera background as texture
- Halo of 3D spheres rotating around head
- Mouth-reactive rectangles

Requirements:
- OpenGL 3.2 Core Profile
- GLFW for windowing
- VBOs cached (geometry created once)
- glLineWidth = 1.0 only
- No fancy effects
"""

import os
import time
import math
from typing import Optional, List, Tuple
import ctypes

import numpy as np
import cv2

try:
    import glfw
    from OpenGL.GL import *
    from OpenGL.GL import shaders as gl_shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

from .face_tracker import FaceTracker, FaceLandmarks
from .metrics import (
    face_width, halo_radius, mouth_openness, face_center,
    halo_sphere_positions, mouth_rect_scale, mouth_rect_color,
    smooth_value, clamp, forehead_center, mouth_center, mouth_width,
    halo_sphere_positions_v2, robot_mouth_bar_color, robot_mouth_bar_heights,
    mouth_plate_dimensions, estimate_face_roll,
    left_eye_openness, right_eye_openness, left_eye_center, right_eye_center,
    left_eye_width, right_eye_width, robot_eye_bar_color, robot_eye_bar_heights,
    eye_plate_dimensions
)
from .primitives import build_sphere, build_quad, build_cube
import random


# MediaPipe FaceMesh connection indices for debug visualization
# These define the lines connecting landmarks for face contour, lips, etc.
FACE_OVAL_PATH = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]
LIPS_PATH = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]
LEFT_EYE_PATH = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE_PATH = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
LEFT_EYEBROW_PATH = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]
RIGHT_EYEBROW_PATH = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300]
NOSE_PATH = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]


# ============================================================================
# Orbiting Cubes with Trails System
# ============================================================================

class CubeData:
    """Individual cube orbital parameters."""

    def __init__(self):
        self.orbit_radius_factor = random.uniform(0.8, 1.8)
        self.orbit_speed = random.uniform(0.4, 1.2)
        self.spin_speed = random.uniform(0.6, 2.5)
        self.wobble_speed = random.uniform(1.4, 3.0)
        self.wobble_amp = random.uniform(0.3, 1.0)
        # Color: vibrant neon colors (cyan, magenta, blue) for better visibility
        color_choice = random.randint(0, 2)
        if color_choice == 0:
            # Intense cyan
            self.color = (0.2, 0.9 + random.uniform(0, 0.1), 1.0)
        elif color_choice == 1:
            # Neon magenta/pink
            self.color = (1.0, 0.3 + random.uniform(0, 0.2), 0.8 + random.uniform(0, 0.2))
        else:
            # Electric blue
            self.color = (0.3 + random.uniform(0, 0.2), 0.5 + random.uniform(0, 0.2), 1.0)


class Particle:
    """Trail particle data."""

    def __init__(self):
        self.pos = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.life = 0.0
        self.color = (1.0, 1.0, 1.0)


class OrbitingCubesSystem:
    """
    System for orbiting cubes with particle trails.

    Manages multiple cubes orbiting around a center point,
    each leaving a trail of fading particles.
    """

    NUM_CUBES = 12
    CUBE_SIZE = 0.017  # Increased from 0.012 for better visibility
    PARTICLE_COUNT = 500  # Increased for denser trails

    def __init__(self):
        self.cubes = [CubeData() for _ in range(self.NUM_CUBES)]
        self.particles = [Particle() for _ in range(self.PARTICLE_COUNT)]
        self.time = 0.0
        self.particle_emit_idx = 0

    def emit_trail_particle(self, cube_pos: Tuple[float, float, float],
                            trail_dir: Tuple[float, float, float],
                            color: Tuple[float, float, float]):
        """Emit a particle from a cube position."""
        p = self.particles[self.particle_emit_idx]
        self.particle_emit_idx = (self.particle_emit_idx + 1) % self.PARTICLE_COUNT

        p.pos = list(cube_pos)
        # Velocity opposite to movement direction
        speed = random.uniform(0.02, 0.05)
        p.vel = [-trail_dir[0] * speed, -trail_dir[1] * speed, -trail_dir[2] * speed]
        p.life = 1.0
        p.color = color

    def update(self, dt: float, center: Tuple[float, float, float],
               base_radius: float):
        """
        Update cubes and particles.

        Args:
            dt: Delta time in seconds
            center: Orbit center position (face center)
            base_radius: Base orbit radius (scaled by face)
        """
        self.time += dt

        # Update each cube and emit particles
        for i, cube in enumerate(self.cubes):
            # Calculate orbital position
            angle = self.time * cube.orbit_speed + (i * 2.0 * math.pi / self.NUM_CUBES)
            orbit_r = base_radius * cube.orbit_radius_factor

            ox = center[0] + orbit_r * math.cos(angle)
            oz = center[2] + orbit_r * math.sin(angle) * 0.4
            oy = center[1] + math.sin(self.time * cube.wobble_speed + i) * cube.wobble_amp * base_radius * 0.3

            # Trail direction (tangent to orbit)
            trail_dir = (
                -math.sin(angle),
                0.0,
                math.cos(angle) * 0.4
            )

            # Emit 2 particles per cube per frame
            for _ in range(2):
                self.emit_trail_particle((ox, oy, oz), trail_dir, cube.color)

        # Update particles
        for p in self.particles:
            if p.life <= 0:
                continue
            p.pos[0] += p.vel[0] * dt
            p.pos[1] += p.vel[1] * dt
            p.pos[2] += p.vel[2] * dt
            p.life -= dt * 0.8  # Slower fade for longer, more visible trails

    def get_cube_transforms(self, center: Tuple[float, float, float],
                            base_radius: float) -> List[Tuple]:
        """
        Get cube positions and rotations for rendering.

        Returns:
            List of (position, spin_y, wobble_x, wobble_z, color) tuples
        """
        transforms = []

        for i, cube in enumerate(self.cubes):
            angle = self.time * cube.orbit_speed + (i * 2.0 * math.pi / self.NUM_CUBES)
            orbit_r = base_radius * cube.orbit_radius_factor

            ox = center[0] + orbit_r * math.cos(angle)
            oz = center[2] + orbit_r * math.sin(angle) * 0.4
            oy = center[1] + math.sin(self.time * cube.wobble_speed + i) * cube.wobble_amp * base_radius * 0.3

            spin_y = self.time * cube.spin_speed
            wobble_x = math.sin(self.time * cube.wobble_speed + i) * 0.4
            wobble_z = math.sin(self.time * cube.wobble_speed * 1.3 + i) * 0.4

            transforms.append(((ox, oy, oz), spin_y, wobble_x, wobble_z, cube.color))

        return transforms

    def get_particle_data(self) -> np.ndarray:
        """
        Get particle data for VBO upload.

        Returns:
            Numpy array of [x, y, z, r, g, b, life] per particle
        """
        data = []
        for p in self.particles:
            if p.life > 0:
                data.extend(p.pos)
                data.extend(p.color)
                data.append(p.life)
            else:
                # Dead particle - zero alpha
                data.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(data, dtype=np.float32)


class ARFilterApp:
    """
    Main AR filter application using OpenGL + GLFW.

    Renders camera feed with overlaid AR effects:
    - Halo of spheres above head
    - Mouth-reactive rectangles
    """

    # Window settings
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    WINDOW_TITLE = "AR Filter - Press ESC to exit"

    # Halo settings
    NUM_HALO_SPHERES = 8
    SPHERE_SIZE = 0.015
    HALO_ROTATION_SPEED = 0.8  # radians per second

    # Mouth rect settings
    NUM_MOUTH_RECTS = 3
    RECT_BASE_WIDTH = 0.02
    RECT_BASE_HEIGHT = 0.015

    def __init__(self, camera_index: int = 0):
        """
        Initialize AR Filter application.

        Args:
            camera_index: OpenCV camera index
        """
        if not OPENGL_AVAILABLE:
            raise RuntimeError("OpenGL/GLFW not available. Install with: pip install PyOpenGL glfw")

        self.camera_index = camera_index
        self.window = None
        self.cap = None
        self.face_tracker = None

        # OpenGL resources
        self.shader_program = None
        self.bg_shader_program = None
        self.bg_texture = None
        self.bg_vao = None
        self.bg_vbo = None

        # Sphere geometry (cached)
        self.sphere_vao = None
        self.sphere_vbo = None
        self.sphere_nbo = None
        self.sphere_ibo = None
        self.sphere_index_count = 0

        # Quad geometry (cached)
        self.quad_vao = None
        self.quad_vbo = None
        self.quad_nbo = None
        self.quad_ibo = None
        self.quad_index_count = 0

        # FaceMesh debug rendering
        self.mesh_shader_program = None
        self.mesh_vao = None
        self.mesh_vbo = None
        self.mesh_vbo_bytes = 0  # Track allocated size for dynamic realloc
        self.show_debug_mesh = True  # Toggle with 'D' key

        # Cube geometry (cached)
        self.cube_vao = None
        self.cube_vbo = None
        self.cube_nbo = None
        self.cube_ibo = None
        self.cube_index_count = 0

        # Particle system
        self.particle_shader_program = None
        self.particle_vao = None
        self.particle_vbo = None

        # Orbiting cubes system
        self.cubes_system = OrbitingCubesSystem()

        # Robot mouth state
        self.robot_mouth_time = 0.0

        # Robot eyes state
        self.robot_eye_time = 0.0
        self.smoothed_left_eye = 0.5
        self.smoothed_right_eye = 0.5

        # Tracking color state (A=yellow, R=red, V=green, B=blue)
        self.tracking_color = (0.2, 0.6, 1.0)  # Default: blue
        self.tracking_color_name = "Blue"

        # State
        self.running = False
        self.halo_angle = 0.0
        self.last_time = 0.0
        self.smoothed_mouth = 0.0
        self.frame_count = 0
        self.fps = 0.0
        self.fps_update_time = 0.0

    def _init_glfw(self) -> bool:
        """Initialize GLFW and create window."""
        if not glfw.init():
            print("ERROR: Failed to initialize GLFW")
            return False

        # Request OpenGL 3.2 Core Profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        # Create window
        self.window = glfw.create_window(
            self.WINDOW_WIDTH, self.WINDOW_HEIGHT,
            self.WINDOW_TITLE, None, None
        )

        if not self.window:
            print("ERROR: Failed to create GLFW window")
            glfw.terminate()
            return False

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # VSync

        # Key callback for ESC
        glfw.set_key_callback(self.window, self._key_callback)

        return True

    def _key_callback(self, window, key, scancode, action, mods):
        """Handle key events."""
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.running = False
        elif key == glfw.KEY_D and action == glfw.PRESS:
            self.show_debug_mesh = not self.show_debug_mesh
            state = "ON" if self.show_debug_mesh else "OFF"
            print(f"[DEBUG] FaceMesh visualization: {state}")
        # Tracking color controls (A/R/V/B)
        elif key == glfw.KEY_A and action == glfw.PRESS:
            self.tracking_color = (1.0, 1.0, 0.2)  # Yellow (Amarillo)
            self.tracking_color_name = "Amarillo"
            print(f"[COLOR] Tracking color: {self.tracking_color_name}")
        elif key == glfw.KEY_R and action == glfw.PRESS:
            self.tracking_color = (1.0, 0.2, 0.2)  # Red (Rojo)
            self.tracking_color_name = "Rojo"
            print(f"[COLOR] Tracking color: {self.tracking_color_name}")
        elif key == glfw.KEY_V and action == glfw.PRESS:
            self.tracking_color = (0.2, 1.0, 0.3)  # Green (Verde)
            self.tracking_color_name = "Verde"
            print(f"[COLOR] Tracking color: {self.tracking_color_name}")
        elif key == glfw.KEY_B and action == glfw.PRESS:
            self.tracking_color = (0.2, 0.6, 1.0)  # Blue (Azul)
            self.tracking_color_name = "Azul"
            print(f"[COLOR] Tracking color: {self.tracking_color_name}")

    def _init_camera(self) -> bool:
        """Initialize OpenCV camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("ERROR: Failed to open camera")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.WINDOW_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return True

    def _load_shaders(self) -> bool:
        """Load and compile shaders."""
        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")

        try:
            # Load 3D object shader
            with open(os.path.join(shader_dir, "basic.vert"), "r") as f:
                vert_src = f.read()
            with open(os.path.join(shader_dir, "basic.frag"), "r") as f:
                frag_src = f.read()

            vert_shader = gl_shaders.compileShader(vert_src, GL_VERTEX_SHADER)
            frag_shader = gl_shaders.compileShader(frag_src, GL_FRAGMENT_SHADER)
            self.shader_program = gl_shaders.compileProgram(vert_shader, frag_shader)

            # Background shader (simple textured quad)
            bg_vert_src = """
            #version 150 core
            in vec2 position;
            in vec2 texCoord;
            out vec2 fragTexCoord;
            void main() {
                fragTexCoord = texCoord;
                gl_Position = vec4(position, 0.0, 1.0);
            }
            """

            bg_frag_src = """
            #version 150 core
            in vec2 fragTexCoord;
            out vec4 outColor;
            uniform sampler2D bgTexture;
            void main() {
                outColor = texture(bgTexture, fragTexCoord);
            }
            """

            bg_vert = gl_shaders.compileShader(bg_vert_src, GL_VERTEX_SHADER)
            bg_frag = gl_shaders.compileShader(bg_frag_src, GL_FRAGMENT_SHADER)
            self.bg_shader_program = gl_shaders.compileProgram(bg_vert, bg_frag)

            # FaceMesh debug shader (simple 2D points/lines)
            mesh_vert_src = """
            #version 150 core
            in vec2 position;
            uniform vec2 screenSize;
            void main() {
                // Convert normalized coords [0,1] to clip space [-1,1]
                // Note: Y is flipped (0 at top in normalized, -1 at bottom in clip)
                vec2 clipPos = position * 2.0 - 1.0;
                clipPos.y = -clipPos.y;  // Flip Y
                gl_Position = vec4(clipPos, 0.0, 1.0);
                gl_PointSize = 3.0;
            }
            """

            mesh_frag_src = """
            #version 150 core
            out vec4 outColor;
            uniform vec3 meshColor;
            void main() {
                outColor = vec4(meshColor, 1.0);
            }
            """

            mesh_vert = gl_shaders.compileShader(mesh_vert_src, GL_VERTEX_SHADER)
            mesh_frag = gl_shaders.compileShader(mesh_frag_src, GL_FRAGMENT_SHADER)
            self.mesh_shader_program = gl_shaders.compileProgram(mesh_vert, mesh_frag)

            # Particle shader for trails
            with open(os.path.join(shader_dir, "particle.vert"), "r") as f:
                part_vert_src = f.read()
            with open(os.path.join(shader_dir, "particle.frag"), "r") as f:
                part_frag_src = f.read()

            part_vert = gl_shaders.compileShader(part_vert_src, GL_VERTEX_SHADER)
            part_frag = gl_shaders.compileShader(part_frag_src, GL_FRAGMENT_SHADER)
            self.particle_shader_program = gl_shaders.compileProgram(part_vert, part_frag)

            return True

        except Exception as e:
            print(f"ERROR: Failed to load shaders: {e}")
            return False

    def _init_geometry(self):
        """Initialize cached geometry (spheres and quads)."""
        # Build sphere geometry
        sphere_verts, sphere_norms, sphere_indices = build_sphere(
            radius=1.0, lat_segments=8, lon_segments=12
        )
        self.sphere_index_count = len(sphere_indices) * 3

        self.sphere_vao = glGenVertexArrays(1)
        glBindVertexArray(self.sphere_vao)

        # Vertices
        self.sphere_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_vbo)
        glBufferData(GL_ARRAY_BUFFER, sphere_verts.nbytes, sphere_verts, GL_STATIC_DRAW)

        pos_loc = glGetAttribLocation(self.shader_program, "position")
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        # Normals
        self.sphere_nbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_nbo)
        glBufferData(GL_ARRAY_BUFFER, sphere_norms.nbytes, sphere_norms, GL_STATIC_DRAW)

        norm_loc = glGetAttribLocation(self.shader_program, "normal")
        glEnableVertexAttribArray(norm_loc)
        glVertexAttribPointer(norm_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        # Indices
        self.sphere_ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.sphere_ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.nbytes, sphere_indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

        # Build quad geometry
        quad_verts, quad_norms, quad_indices = build_quad(width=1.0, height=1.0)
        self.quad_index_count = len(quad_indices) * 3

        self.quad_vao = glGenVertexArrays(1)
        glBindVertexArray(self.quad_vao)

        self.quad_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_verts.nbytes, quad_verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        self.quad_nbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_nbo)
        glBufferData(GL_ARRAY_BUFFER, quad_norms.nbytes, quad_norms, GL_STATIC_DRAW)
        glEnableVertexAttribArray(norm_loc)
        glVertexAttribPointer(norm_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        self.quad_ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quad_ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

        # Background quad (fullscreen)
        bg_data = np.array([
            # position   texcoord
            -1.0, -1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 0.0,
        ], dtype=np.float32)

        self.bg_vao = glGenVertexArrays(1)
        glBindVertexArray(self.bg_vao)

        self.bg_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.bg_vbo)
        glBufferData(GL_ARRAY_BUFFER, bg_data.nbytes, bg_data, GL_STATIC_DRAW)

        bg_pos_loc = glGetAttribLocation(self.bg_shader_program, "position")
        bg_tex_loc = glGetAttribLocation(self.bg_shader_program, "texCoord")

        glEnableVertexAttribArray(bg_pos_loc)
        glVertexAttribPointer(bg_pos_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))

        glEnableVertexAttribArray(bg_tex_loc)
        glVertexAttribPointer(bg_tex_loc, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))

        glBindVertexArray(0)

        # Background texture
        self.bg_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.bg_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # FaceMesh debug geometry (dynamic VBO - size allocated on first use)
        self.mesh_vao = glGenVertexArrays(1)
        glBindVertexArray(self.mesh_vao)

        self.mesh_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh_vbo)
        # Pre-allocate for 478 landmarks (refine_landmarks=True includes iris)
        initial_size = 478 * 2 * 4
        glBufferData(GL_ARRAY_BUFFER, initial_size, None, GL_DYNAMIC_DRAW)
        self.mesh_vbo_bytes = initial_size

        mesh_pos_loc = glGetAttribLocation(self.mesh_shader_program, "position")
        glEnableVertexAttribArray(mesh_pos_loc)
        glVertexAttribPointer(mesh_pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)

        glBindVertexArray(0)

        # Build cube geometry for orbiting cubes
        cube_verts, cube_norms, cube_indices = build_cube(size=1.0)
        self.cube_index_count = len(cube_indices) * 3

        self.cube_vao = glGenVertexArrays(1)
        glBindVertexArray(self.cube_vao)

        self.cube_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_vbo)
        glBufferData(GL_ARRAY_BUFFER, cube_verts.nbytes, cube_verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        self.cube_nbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cube_nbo)
        glBufferData(GL_ARRAY_BUFFER, cube_norms.nbytes, cube_norms, GL_STATIC_DRAW)
        glEnableVertexAttribArray(norm_loc)
        glVertexAttribPointer(norm_loc, 3, GL_FLOAT, GL_FALSE, 0, None)

        self.cube_ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.cube_ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube_indices.nbytes, cube_indices, GL_STATIC_DRAW)

        glBindVertexArray(0)

        # Particle VBO for trails (dynamic)
        self.particle_vao = glGenVertexArrays(1)
        glBindVertexArray(self.particle_vao)

        self.particle_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        # Pre-allocate for particles: 7 floats per particle (x,y,z,r,g,b,life)
        particle_buffer_size = OrbitingCubesSystem.PARTICLE_COUNT * 7 * 4
        glBufferData(GL_ARRAY_BUFFER, particle_buffer_size, None, GL_DYNAMIC_DRAW)

        # Particle attributes: position (3), color (3), life (1)
        part_pos_loc = glGetAttribLocation(self.particle_shader_program, "position")
        part_color_loc = glGetAttribLocation(self.particle_shader_program, "color")
        part_life_loc = glGetAttribLocation(self.particle_shader_program, "life")

        stride = 7 * 4  # 7 floats * 4 bytes
        glEnableVertexAttribArray(part_pos_loc)
        glVertexAttribPointer(part_pos_loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        glEnableVertexAttribArray(part_color_loc)
        glVertexAttribPointer(part_color_loc, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        glEnableVertexAttribArray(part_life_loc)
        glVertexAttribPointer(part_life_loc, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

        glBindVertexArray(0)

    def _create_projection_matrix(self) -> np.ndarray:
        """Create orthographic projection matrix."""
        # Simple orthographic for 2D-like rendering
        left, right = 0.0, 1.0
        bottom, top = 1.0, 0.0  # Flipped for screen coords
        near, far = -1.0, 1.0

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2.0 / (right - left)
        proj[1, 1] = 2.0 / (top - bottom)
        proj[2, 2] = -2.0 / (far - near)
        proj[0, 3] = -(right + left) / (right - left)
        proj[1, 3] = -(top + bottom) / (top - bottom)
        proj[2, 3] = -(far + near) / (far - near)
        proj[3, 3] = 1.0

        return proj

    def _create_view_matrix(self) -> np.ndarray:
        """Create identity view matrix."""
        return np.eye(4, dtype=np.float32)

    def _create_model_matrix(self, position: Tuple[float, float, float],
                            scale: float) -> np.ndarray:
        """Create model matrix with position and uniform scale."""
        model = np.eye(4, dtype=np.float32)

        # Scale
        model[0, 0] = scale
        model[1, 1] = scale
        model[2, 2] = scale

        # Translation
        model[0, 3] = position[0]
        model[1, 3] = position[1]
        model[2, 3] = position[2]

        return model

    def _update_background_texture(self, frame):
        """Update background texture with camera frame."""
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        glBindTexture(GL_TEXTURE_2D, self.bg_texture)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            rgb_frame.shape[1], rgb_frame.shape[0],
            0, GL_RGB, GL_UNSIGNED_BYTE, rgb_frame
        )

    def _render_background(self):
        """Render camera background."""
        glUseProgram(self.bg_shader_program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.bg_texture)
        tex_loc = glGetUniformLocation(self.bg_shader_program, "bgTexture")
        glUniform1i(tex_loc, 0)

        glBindVertexArray(self.bg_vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

    def _render_sphere(self, position: Tuple[float, float, float],
                       scale: float, color: Tuple[float, float, float]):
        """Render a sphere at given position."""
        model = self._create_model_matrix(position, scale)
        view = self._create_view_matrix()
        proj = self._create_projection_matrix()

        glUseProgram(self.shader_program)

        # Set uniforms
        model_loc = glGetUniformLocation(self.shader_program, "model")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        color_loc = glGetUniformLocation(self.shader_program, "objectColor")
        light_loc = glGetUniformLocation(self.shader_program, "lightDir")
        ambient_loc = glGetUniformLocation(self.shader_program, "ambient")

        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj)
        glUniform3f(color_loc, *color)
        glUniform3f(light_loc, 0.5, -0.5, 1.0)
        glUniform1f(ambient_loc, 0.4)

        glBindVertexArray(self.sphere_vao)
        glDrawElements(GL_TRIANGLES, self.sphere_index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def _render_quad(self, position: Tuple[float, float, float],
                     width: float, height: float,
                     color: Tuple[float, float, float]):
        """Render a quad at given position."""
        # Scale for width/height
        model = np.eye(4, dtype=np.float32)
        model[0, 0] = width
        model[1, 1] = height
        model[0, 3] = position[0]
        model[1, 3] = position[1]
        model[2, 3] = position[2]

        view = self._create_view_matrix()
        proj = self._create_projection_matrix()

        glUseProgram(self.shader_program)

        model_loc = glGetUniformLocation(self.shader_program, "model")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        color_loc = glGetUniformLocation(self.shader_program, "objectColor")
        light_loc = glGetUniformLocation(self.shader_program, "lightDir")
        ambient_loc = glGetUniformLocation(self.shader_program, "ambient")

        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj)
        glUniform3f(color_loc, *color)
        glUniform3f(light_loc, 0.0, 0.0, 1.0)
        glUniform1f(ambient_loc, 0.6)

        glBindVertexArray(self.quad_vao)
        glDrawElements(GL_TRIANGLES, self.quad_index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def _create_rotation_matrix_y(self, angle: float) -> np.ndarray:
        """Create Y-axis rotation matrix."""
        c, s = math.cos(angle), math.sin(angle)
        rot = np.eye(4, dtype=np.float32)
        rot[0, 0] = c
        rot[0, 2] = s
        rot[2, 0] = -s
        rot[2, 2] = c
        return rot

    def _create_rotation_matrix_x(self, angle: float) -> np.ndarray:
        """Create X-axis rotation matrix."""
        c, s = math.cos(angle), math.sin(angle)
        rot = np.eye(4, dtype=np.float32)
        rot[1, 1] = c
        rot[1, 2] = -s
        rot[2, 1] = s
        rot[2, 2] = c
        return rot

    def _create_rotation_matrix_z(self, angle: float) -> np.ndarray:
        """Create Z-axis rotation matrix."""
        c, s = math.cos(angle), math.sin(angle)
        rot = np.eye(4, dtype=np.float32)
        rot[0, 0] = c
        rot[0, 1] = -s
        rot[1, 0] = s
        rot[1, 1] = c
        return rot

    def _render_cube(self, position: Tuple[float, float, float],
                     scale: float, color: Tuple[float, float, float],
                     spin_y: float = 0.0, wobble_x: float = 0.0, wobble_z: float = 0.0):
        """Render a cube with rotation at given position."""
        # Build model matrix: translate * rotY * rotX * rotZ * scale
        trans = np.eye(4, dtype=np.float32)
        trans[0, 3] = position[0]
        trans[1, 3] = position[1]
        trans[2, 3] = position[2]

        scale_mat = np.eye(4, dtype=np.float32)
        scale_mat[0, 0] = scale
        scale_mat[1, 1] = scale
        scale_mat[2, 2] = scale

        rot_y = self._create_rotation_matrix_y(spin_y)
        rot_x = self._create_rotation_matrix_x(wobble_x)
        rot_z = self._create_rotation_matrix_z(wobble_z)

        # Combine: trans * rotY * rotX * rotZ * scale
        model = trans @ rot_y @ rot_x @ rot_z @ scale_mat

        view = self._create_view_matrix()
        proj = self._create_projection_matrix()

        glUseProgram(self.shader_program)

        model_loc = glGetUniformLocation(self.shader_program, "model")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        color_loc = glGetUniformLocation(self.shader_program, "objectColor")
        light_loc = glGetUniformLocation(self.shader_program, "lightDir")
        ambient_loc = glGetUniformLocation(self.shader_program, "ambient")

        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj)
        glUniform3f(color_loc, *color)
        glUniform3f(light_loc, 0.5, -0.5, 1.0)
        glUniform1f(ambient_loc, 0.35)

        glBindVertexArray(self.cube_vao)
        glDrawElements(GL_TRIANGLES, self.cube_index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def _render_particles(self):
        """Render particle trails."""
        particle_data = self.cubes_system.get_particle_data()

        glUseProgram(self.particle_shader_program)

        view = self._create_view_matrix()
        proj = self._create_projection_matrix()

        view_loc = glGetUniformLocation(self.particle_shader_program, "view")
        proj_loc = glGetUniformLocation(self.particle_shader_program, "projection")

        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view)
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, proj)

        # Enable blending for alpha
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_PROGRAM_POINT_SIZE)

        glBindBuffer(GL_ARRAY_BUFFER, self.particle_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, particle_data.nbytes, particle_data)

        glBindVertexArray(self.particle_vao)
        glDrawArrays(GL_POINTS, 0, OrbitingCubesSystem.PARTICLE_COUNT)
        glBindVertexArray(0)

        glDisable(GL_PROGRAM_POINT_SIZE)
        glDisable(GL_BLEND)

    def _render_facemesh_debug(self, landmarks: List[Tuple[float, float, float]]):
        """
        Render FaceMesh landmarks for debugging.

        Draws:
        - All landmarks as points (green) - supports 468 or 478 (with iris)
        - Face contour as lines (white)
        - Lips as lines (cyan)
        - Eyes and eyebrows as lines (yellow)
        """
        if not landmarks or len(landmarks) < 468:
            return

        glUseProgram(self.mesh_shader_program)

        # Enable point size (for GL_POINTS)
        glEnable(GL_PROGRAM_POINT_SIZE)

        # Update VBO with current landmark positions
        # Convert to 2D array: [(x1, y1), (x2, y2), ...]
        # Need to flip X because camera is mirrored
        points_2d = np.array(
            [(1.0 - lm[0], lm[1]) for lm in landmarks],
            dtype=np.float32
        )
        # Ensure contiguous memory layout for OpenGL
        points_2d = np.ascontiguousarray(points_2d)

        glBindBuffer(GL_ARRAY_BUFFER, self.mesh_vbo)

        # Reallocate VBO if landmark count changed (468 vs 478 with iris)
        needed_bytes = points_2d.nbytes
        if needed_bytes > self.mesh_vbo_bytes:
            glBufferData(GL_ARRAY_BUFFER, needed_bytes, None, GL_DYNAMIC_DRAW)
            self.mesh_vbo_bytes = needed_bytes

        glBufferSubData(GL_ARRAY_BUFFER, 0, needed_bytes, points_2d)

        color_loc = glGetUniformLocation(self.mesh_shader_program, "meshColor")

        glBindVertexArray(self.mesh_vao)

        # Draw all points in green
        glUniform3f(color_loc, 0.0, 1.0, 0.0)
        glDrawArrays(GL_POINTS, 0, len(landmarks))

        # Draw face oval contour in white
        glUniform3f(color_loc, 1.0, 1.0, 1.0)
        self._draw_landmark_path(points_2d, FACE_OVAL_PATH)

        # Draw lips in cyan
        glUniform3f(color_loc, 0.0, 1.0, 1.0)
        self._draw_landmark_path(points_2d, LIPS_PATH)

        # Draw eyes in yellow
        glUniform3f(color_loc, 1.0, 1.0, 0.0)
        self._draw_landmark_path(points_2d, LEFT_EYE_PATH)
        self._draw_landmark_path(points_2d, RIGHT_EYE_PATH)

        # Draw eyebrows in orange
        glUniform3f(color_loc, 1.0, 0.6, 0.0)
        self._draw_landmark_path(points_2d, LEFT_EYEBROW_PATH)
        self._draw_landmark_path(points_2d, RIGHT_EYEBROW_PATH)

        # Draw nose in magenta
        glUniform3f(color_loc, 1.0, 0.0, 1.0)
        self._draw_landmark_path(points_2d, NOSE_PATH)

        glBindVertexArray(0)
        glDisable(GL_PROGRAM_POINT_SIZE)

    def _draw_landmark_path(self, points_2d: np.ndarray, indices: List[int]):
        """Draw a path connecting landmarks by indices."""
        if len(indices) < 2:
            return

        # Create line strip data
        path_points = np.array(
            [points_2d[i] for i in indices if i < len(points_2d)],
            dtype=np.float32
        )

        if len(path_points) < 2:
            return

        # Upload and draw
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, path_points.nbytes, path_points)
        glDrawArrays(GL_LINE_STRIP, 0, len(path_points))

    def _render_halo(self, landmarks: List[Tuple[float, float, float]], dt: float):
        """Render orbiting cubes with particle trails around the head."""
        # Get face metrics
        fw = face_width(landmarks)
        if fw < 0.01:
            return

        # Base orbit radius scaled by face width
        base_radius = halo_radius(fw, scale_factor=0.9)
        forehead = forehead_center(landmarks)

        # Mirror X coordinate to match the flipped camera
        # Position orbit center above forehead
        orbit_center = (
            1.0 - forehead[0],
            forehead[1] - 0.05,  # Slightly above forehead
            forehead[2]
        )

        # Update cubes system
        self.cubes_system.update(dt, orbit_center, base_radius)

        # Render particles first (behind cubes)
        self._render_particles()

        # Get cube transforms and render each cube
        transforms = self.cubes_system.get_cube_transforms(orbit_center, base_radius)
        for pos, spin_y, wobble_x, wobble_z, color in transforms:
            self._render_cube(
                pos, OrbitingCubesSystem.CUBE_SIZE, color,
                spin_y, wobble_x, wobble_z
            )

    def _render_mouth_rects(self, landmarks: List[Tuple[float, float, float]], dt: float):
        """
        Render robot mouth that covers real mouth with animated bars.

        Features:
        - Dark plate covering the real mouth
        - N animated bars that react to mouth openness
        - Smooth color gradient (blue → yellow → red)
        - Pulsing animation when speaking
        """
        # Update time for animations
        self.robot_mouth_time += dt

        # Get mouth openness with smoothing (alpha=0.2 for stability)
        raw_mouth = mouth_openness(landmarks)
        self.smoothed_mouth = smooth_value(self.smoothed_mouth, raw_mouth, 0.2)

        # Get mouth position and dimensions
        mouth_pos = mouth_center(landmarks)
        plate_width, plate_height = mouth_plate_dimensions(landmarks, width_scale=1.3, height_ratio=0.5)

        # Get face roll for slight rotation with mouth
        roll = estimate_face_roll(landmarks)

        # Mirror X coordinate to match the flipped camera
        mouth_pos_mirrored = (1.0 - mouth_pos[0], mouth_pos[1], mouth_pos[2])

        # Enable blending for alpha
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 1. Render dark plate covering the mouth
        plate_color = (0.08, 0.08, 0.12)  # Dark blue-gray
        self._render_quad(
            (mouth_pos_mirrored[0], mouth_pos_mirrored[1], 0.05),
            plate_width, plate_height, plate_color
        )

        # 2. Render optional neon border (glow effect with two layers)
        border_color_outer = (0.2, 0.6, 1.0)  # Cyan glow outer
        border_color_inner = (0.4, 0.8, 1.0)  # Brighter inner

        # Outer glow border (slightly larger)
        self._render_quad(
            (mouth_pos_mirrored[0], mouth_pos_mirrored[1], 0.04),
            plate_width * 1.08, plate_height * 1.15,
            (border_color_outer[0] * 0.3, border_color_outer[1] * 0.3, border_color_outer[2] * 0.3)
        )

        # 3. Render animated bars
        num_bars = 7
        bar_heights = robot_mouth_bar_heights(
            num_bars, self.smoothed_mouth, self.robot_mouth_time,
            base_height=0.25, max_height=1.0, pulse_freq=10.0
        )
        bar_color = robot_mouth_bar_color(self.smoothed_mouth)

        # Calculate bar dimensions
        total_bar_width = plate_width * 0.85
        bar_spacing = total_bar_width / num_bars
        bar_width = bar_spacing * 0.7
        max_bar_height = plate_height * 0.8

        # Start position (left side of bars)
        start_x = mouth_pos_mirrored[0] - total_bar_width / 2 + bar_spacing / 2

        for i in range(num_bars):
            # Bar position
            bar_x = start_x + i * bar_spacing
            bar_h = bar_heights[i] * max_bar_height

            # Render bar
            self._render_quad(
                (bar_x, mouth_pos_mirrored[1], 0.06),
                bar_width, bar_h, bar_color
            )

            # Add highlight on top of each bar
            highlight_color = (
                min(1.0, bar_color[0] + 0.3),
                min(1.0, bar_color[1] + 0.3),
                min(1.0, bar_color[2] + 0.3)
            )
            self._render_quad(
                (bar_x, mouth_pos_mirrored[1] - bar_h * 0.4, 0.07),
                bar_width * 0.6, bar_h * 0.15, highlight_color
            )

        glDisable(GL_BLEND)

    def _render_robot_eyes(self, landmarks: List[Tuple[float, float, float]], dt: float):
        """
        Render robotic animated eyes with semi-circular plates and animated bars.

        Features:
        - Semi-circular plate over each eye
        - N animated vertical bars per eye reacting to eye openness/blinks
        - Smooth color gradient (dark blue → cyan → yellow → orange)
        - Fast nervous pulse animation
        """
        # Update time for animations
        self.robot_eye_time += dt

        # Get eye openness with smoothing (alpha=0.2)
        raw_left_eye = left_eye_openness(landmarks)
        raw_right_eye = right_eye_openness(landmarks)
        self.smoothed_left_eye = smooth_value(self.smoothed_left_eye, raw_left_eye, 0.2)
        self.smoothed_right_eye = smooth_value(self.smoothed_right_eye, raw_right_eye, 0.2)

        # Get eye positions and dimensions
        left_eye_pos = left_eye_center(landmarks)
        right_eye_pos = right_eye_center(landmarks)
        left_width = left_eye_width(landmarks)
        right_width = right_eye_width(landmarks)

        # Calculate plate dimensions for each eye
        left_plate_w, left_plate_h = eye_plate_dimensions(landmarks, is_left=True, width_scale=1.8)
        right_plate_w, right_plate_h = eye_plate_dimensions(landmarks, is_left=False, width_scale=1.8)

        # Mirror X coordinates to match the flipped camera
        left_eye_mirrored = (1.0 - left_eye_pos[0], left_eye_pos[1], left_eye_pos[2])
        right_eye_mirrored = (1.0 - right_eye_pos[0], right_eye_pos[1], right_eye_pos[2])

        # Enable blending for alpha
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Render both eyes
        for eye_pos, plate_w, plate_h, eye_openness, eye_name in [
            (left_eye_mirrored, left_plate_w, left_plate_h, self.smoothed_left_eye, "left"),
            (right_eye_mirrored, right_plate_w, right_plate_h, self.smoothed_right_eye, "right")
        ]:
            # 1. Render dark plate covering the eye (semi-circular effect via quad)
            plate_color = (0.05, 0.08, 0.15)  # Dark blue
            self._render_quad(
                (eye_pos[0], eye_pos[1], 0.05),
                plate_w, plate_h, plate_color
            )

            # 2. Render neon border glow
            border_color = (0.1, 0.4, 0.7)  # Cyan glow
            self._render_quad(
                (eye_pos[0], eye_pos[1], 0.04),
                plate_w * 1.1, plate_h * 1.15,
                (border_color[0] * 0.4, border_color[1] * 0.4, border_color[2] * 0.4)
            )

            # 3. Render animated bars
            num_bars = 5
            bar_heights = robot_eye_bar_heights(
                num_bars, eye_openness, self.robot_eye_time,
                base_height=0.3, max_height=1.0, pulse_freq=15.0
            )
            bar_color = robot_eye_bar_color(eye_openness)

            # Calculate bar dimensions
            total_bar_width = plate_w * 0.8
            bar_spacing = total_bar_width / num_bars
            bar_width = bar_spacing * 0.65
            max_bar_height = plate_h * 0.75

            # Start position (left side of bars)
            start_x = eye_pos[0] - total_bar_width / 2 + bar_spacing / 2

            for i in range(num_bars):
                # Bar position
                bar_x = start_x + i * bar_spacing
                bar_h = bar_heights[i] * max_bar_height

                # Render bar
                self._render_quad(
                    (bar_x, eye_pos[1], 0.06),
                    bar_width, bar_h, bar_color
                )

                # Add highlight on top of each bar
                highlight_color = (
                    min(1.0, bar_color[0] + 0.25),
                    min(1.0, bar_color[1] + 0.25),
                    min(1.0, bar_color[2] + 0.25)
                )
                self._render_quad(
                    (bar_x, eye_pos[1] - bar_h * 0.35, 0.07),
                    bar_width * 0.5, bar_h * 0.12, highlight_color
                )

        glDisable(GL_BLEND)

    def run(self):
        """Main application loop."""
        # Initialize
        if not self._init_glfw():
            return

        if not self._init_camera():
            glfw.terminate()
            return

        if not self._load_shaders():
            self.cap.release()
            glfw.terminate()
            return

        self._init_geometry()
        self.face_tracker = FaceTracker()

        # OpenGL settings
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glLineWidth(1.0)  # Required by spec

        self.running = True
        self.last_time = time.time()
        self.fps_update_time = self.last_time

        print("AR Filter started.")
        print("  [ESC] Exit")
        print("  [D]   Toggle FaceMesh debug visualization")
        print("  [A]   Tracking color: Amarillo (Yellow)")
        print("  [R]   Tracking color: Rojo (Red)")
        print("  [V]   Tracking color: Verde (Green)")
        print("  [B]   Tracking color: Azul (Blue)")
        print(f"  Current tracking color: {self.tracking_color_name}")

        while self.running and not glfw.window_should_close(self.window):
            # Calculate delta time
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time

            # FPS counter
            self.frame_count += 1
            if current_time - self.fps_update_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.fps_update_time)
                self.frame_count = 0
                self.fps_update_time = current_time
                glfw.set_window_title(self.window,
                    f"{self.WINDOW_TITLE} | FPS: {self.fps:.1f}")

            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Process face
            landmarks = self.face_tracker.process_frame(frame)

            # Clear
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Render background
            self._update_background_texture(frame)
            glDisable(GL_DEPTH_TEST)
            self._render_background()
            glEnable(GL_DEPTH_TEST)

            # Render AR elements if face detected
            if landmarks:
                # Debug: render FaceMesh visualization first (behind AR elements)
                if self.show_debug_mesh:
                    glDisable(GL_DEPTH_TEST)
                    self._render_facemesh_debug(landmarks)
                    glEnable(GL_DEPTH_TEST)

                # Render AR filter elements
                self._render_halo(landmarks, dt)
                self._render_mouth_rects(landmarks, dt)
                self._render_robot_eyes(landmarks, dt)

            # Swap buffers and poll events
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        # Cleanup
        self._cleanup()

    def _cleanup(self):
        """Release all resources."""
        if self.face_tracker:
            self.face_tracker.release()

        if self.cap:
            self.cap.release()

        # Delete OpenGL resources
        if self.shader_program:
            glDeleteProgram(self.shader_program)
        if self.bg_shader_program:
            glDeleteProgram(self.bg_shader_program)
        if self.mesh_shader_program:
            glDeleteProgram(self.mesh_shader_program)
        if self.particle_shader_program:
            glDeleteProgram(self.particle_shader_program)

        if self.sphere_vao:
            glDeleteVertexArrays(1, [self.sphere_vao])
        if self.mesh_vao:
            glDeleteVertexArrays(1, [self.mesh_vao])
        if self.quad_vao:
            glDeleteVertexArrays(1, [self.quad_vao])
        if self.bg_vao:
            glDeleteVertexArrays(1, [self.bg_vao])
        if self.cube_vao:
            glDeleteVertexArrays(1, [self.cube_vao])
        if self.particle_vao:
            glDeleteVertexArrays(1, [self.particle_vao])

        if self.bg_texture:
            glDeleteTextures(1, [self.bg_texture])

        glfw.terminate()
        print("AR Filter closed.")


def run_ar_filter(camera_index: int = 0):
    """
    Entry point to run the AR filter.

    Args:
        camera_index: OpenCV camera index
    """
    app = ARFilterApp(camera_index=camera_index)
    app.run()


if __name__ == "__main__":
    run_ar_filter()
