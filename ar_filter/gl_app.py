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
    halo_sphere_positions_v2
)
from .primitives import build_sphere, build_quad


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
        self.show_debug_mesh = True  # Toggle with 'D' key

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

        # FaceMesh debug geometry (dynamic VBO)
        self.mesh_vao = glGenVertexArrays(1)
        glBindVertexArray(self.mesh_vao)

        self.mesh_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.mesh_vbo)
        # Pre-allocate for 468 landmarks * 2 floats (x, y)
        glBufferData(GL_ARRAY_BUFFER, 468 * 2 * 4, None, GL_DYNAMIC_DRAW)

        mesh_pos_loc = glGetAttribLocation(self.mesh_shader_program, "position")
        glEnableVertexAttribArray(mesh_pos_loc)
        glVertexAttribPointer(mesh_pos_loc, 2, GL_FLOAT, GL_FALSE, 0, None)

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

    def _render_facemesh_debug(self, landmarks: List[Tuple[float, float, float]]):
        """
        Render FaceMesh landmarks for debugging.

        Draws:
        - All 468 landmarks as points (green)
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

        glBindBuffer(GL_ARRAY_BUFFER, self.mesh_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, points_2d.nbytes, points_2d)

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
        """Render halo of spheres above head using forehead landmark."""
        # Get face metrics
        fw = face_width(landmarks)
        if fw < 0.01:
            return

        hr = halo_radius(fw, scale_factor=0.8)  # Smaller radius for tighter halo
        forehead = forehead_center(landmarks)

        # Mirror X coordinate to match the flipped camera
        forehead_mirrored = (1.0 - forehead[0], forehead[1], forehead[2])

        # Update rotation
        self.halo_angle += self.HALO_ROTATION_SPEED * dt

        # Get sphere positions using forehead-based positioning
        positions = halo_sphere_positions_v2(
            forehead_mirrored, hr, self.NUM_HALO_SPHERES, self.halo_angle,
            height_offset=0.06  # Position above forehead
        )

        # Render each sphere with gradient colors
        for i, pos in enumerate(positions):
            # Color gradient from gold to orange
            t = i / max(1, self.NUM_HALO_SPHERES - 1)
            r = 1.0
            g = 0.8 - t * 0.3
            b = 0.2 + t * 0.2

            self._render_sphere(pos, self.SPHERE_SIZE, (r, g, b))

    def _render_mouth_rects(self, landmarks: List[Tuple[float, float, float]]):
        """Render mouth-reactive rectangles at the actual mouth position."""
        # Get mouth openness
        raw_mouth = mouth_openness(landmarks)
        self.smoothed_mouth = smooth_value(self.smoothed_mouth, raw_mouth, 0.3)

        # Get actual mouth center and width
        mouth_pos = mouth_center(landmarks)
        m_width = mouth_width(landmarks)

        # Mirror X coordinate to match the flipped camera
        mouth_pos_mirrored = (1.0 - mouth_pos[0], mouth_pos[1], mouth_pos[2])

        # Calculate scale and color based on mouth openness
        scale = mouth_rect_scale(self.smoothed_mouth, min_scale=0.5, max_scale=2.5)
        color = mouth_rect_color(self.smoothed_mouth)

        # Base rectangle size proportional to mouth width
        base_width = m_width * 0.25
        base_height = base_width * 0.5 * scale

        # Render rectangles centered on the mouth
        for i in range(self.NUM_MOUTH_RECTS):
            # Offset each rect horizontally, spread across mouth width
            offset_x = (i - 1) * base_width * 1.2

            pos = (
                mouth_pos_mirrored[0] + offset_x,
                mouth_pos_mirrored[1],
                0.1  # Slightly in front
            )

            self._render_quad(pos, base_width, base_height, color)

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
                self._render_mouth_rects(landmarks)

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

        if self.sphere_vao:
            glDeleteVertexArrays(1, [self.sphere_vao])
        if self.mesh_vao:
            glDeleteVertexArrays(1, [self.mesh_vao])
        if self.quad_vao:
            glDeleteVertexArrays(1, [self.quad_vao])
        if self.bg_vao:
            glDeleteVertexArrays(1, [self.bg_vao])

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
