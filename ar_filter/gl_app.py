"""
GL App Module - Main OpenGL application for AR filter

This is the complete standalone OpenGL application.
It initializes GLFW, creates window, loads shaders, and renders.

Does NOT import main.py or any existing project modules.
ESC closes and releases camera.
"""

import os
import sys
import time
import math
import numpy as np
import cv2

# OpenGL imports with fallback
try:
    import glfw
    from OpenGL.GL import *
    from OpenGL.GL import shaders
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

# Local imports (relative, within ar_filter module only)
from .face_tracker import FaceTracker
from .metrics import (
    face_width, face_height, mouth_openness, head_tilt,
    face_center, smooth_value
)
from .primitives import (
    build_circle, build_horn, build_mask_outline,
    build_neon_lines, build_zigzag_line, build_star
)


class NeonMaskFilter:
    """
    Neon Mask AR Filter using OpenGL.

    Visual: Geometric mask over face with animated neon colors.
    Interaction: Mouth open = more intense animation.
    """

    # Neon color palette (RGB normalized 0-1)
    COLORS = {
        'cyan': (0.0, 1.0, 1.0),
        'magenta': (1.0, 0.0, 1.0),
        'yellow': (1.0, 1.0, 0.0),
        'green': (0.0, 1.0, 0.5),
        'blue': (0.2, 0.5, 1.0),
        'pink': (1.0, 0.4, 0.7),
        'orange': (1.0, 0.5, 0.0),
    }

    def __init__(self, width: int = 1280, height: int = 720):
        """
        Initialize the filter application.

        Args:
            width: Window/camera width
            height: Window/camera height
        """
        self.width = width
        self.height = height

        # OpenGL resources
        self.window = None
        self.shader_program = None
        self.vao = None
        self.vbo = None

        # Shader uniform locations
        self.u_projection = None
        self.u_offset = None
        self.u_scale = None
        self.u_color = None
        self.u_alpha = None
        self.u_time = None
        self.u_pulse = None

        # Camera
        self.cap = None

        # Face tracking
        self.face_tracker = FaceTracker()

        # Animation state
        self.start_time = time.time()
        self.current_color_idx = 0
        self.color_names = list(self.COLORS.keys())
        self.smooth_mouth = 0.0
        self.smooth_tilt = 0.0

        # Background texture
        self.bg_texture = None

    def _get_shader_path(self, filename: str) -> str:
        """Get absolute path to shader file."""
        module_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(module_dir, 'shaders', filename)

    def _load_shader_source(self, filename: str) -> str:
        """Load shader source from file."""
        path = self._get_shader_path(filename)
        with open(path, 'r') as f:
            return f.read()

    def _create_shaders(self) -> bool:
        """Compile and link shaders."""
        try:
            vert_source = self._load_shader_source('basic.vert')
            frag_source = self._load_shader_source('basic.frag')

            # Compile vertex shader
            vertex_shader = shaders.compileShader(vert_source, GL_VERTEX_SHADER)

            # Compile fragment shader
            fragment_shader = shaders.compileShader(frag_source, GL_FRAGMENT_SHADER)

            # Link program
            self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)

            # Get uniform locations
            self.u_projection = glGetUniformLocation(self.shader_program, "uProjection")
            self.u_offset = glGetUniformLocation(self.shader_program, "uOffset")
            self.u_scale = glGetUniformLocation(self.shader_program, "uScale")
            self.u_color = glGetUniformLocation(self.shader_program, "uColor")
            self.u_alpha = glGetUniformLocation(self.shader_program, "uAlpha")
            self.u_time = glGetUniformLocation(self.shader_program, "uTime")
            self.u_pulse = glGetUniformLocation(self.shader_program, "uPulse")

            return True

        except Exception as e:
            print(f"[AR FILTER] Shader compilation error: {e}")
            return False

    def _setup_gl_buffers(self):
        """Setup VAO and VBO for dynamic drawing."""
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Position attribute (2 floats per vertex)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, None)
        glEnableVertexAttribArray(0)

        glBindVertexArray(0)

    def _setup_background_texture(self):
        """Create texture for camera background."""
        self.bg_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.bg_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def _create_ortho_projection(self) -> np.ndarray:
        """Create orthographic projection matrix (normalized coords 0-1)."""
        # Map [0,1] to [-1,1] for OpenGL
        proj = np.array([
            [2.0, 0.0, 0.0, -1.0],
            [0.0, -2.0, 0.0, 1.0],  # Flip Y
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        return proj

    def initialize(self) -> bool:
        """
        Initialize GLFW, OpenGL, and camera.

        Returns:
            True if successful, False otherwise
        """
        if not OPENGL_AVAILABLE:
            print("[AR FILTER] OpenGL/GLFW not available!")
            return False

        print("\n" + "=" * 50)
        print("  NEON MASK AR FILTER v1.0")
        print("  Press ESC to exit")
        print("=" * 50 + "\n")

        # Initialize GLFW
        if not glfw.init():
            print("[AR FILTER] Failed to initialize GLFW")
            return False

        # Window hints for OpenGL 3.3 core
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        # Create window
        self.window = glfw.create_window(
            self.width, self.height,
            "Neon Mask AR Filter",
            None, None
        )

        if not self.window:
            print("[AR FILTER] Failed to create window")
            glfw.terminate()
            return False

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # VSync

        # Setup OpenGL
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(2.0)

        # Create shaders
        if not self._create_shaders():
            return False

        # Setup buffers
        self._setup_gl_buffers()
        self._setup_background_texture()

        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("[AR FILTER] Failed to open camera")
            return False

        print("[AR FILTER] Initialization complete!")
        return True

    def _draw_vertices(self, vertices: np.ndarray, mode: int,
                       color: tuple, alpha: float = 1.0, pulse: float = 0.0):
        """
        Draw vertices with current shader settings.

        Args:
            vertices: numpy array of (x, y) vertices
            mode: GL draw mode (GL_LINE_STRIP, GL_LINE_LOOP, GL_TRIANGLES, etc.)
            color: RGB tuple (0-1 range)
            alpha: Opacity (0-1)
            pulse: Pulse intensity (0-1)
        """
        if len(vertices) == 0:
            return

        # Update VBO data
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)

        # Set uniforms
        glUniform3f(self.u_color, *color)
        glUniform1f(self.u_alpha, alpha)
        glUniform1f(self.u_pulse, pulse)

        # Draw
        glDrawArrays(mode, 0, len(vertices))

        glBindVertexArray(0)

    def _render_neon_mask(self, face_data: dict, t: float, intensity: float):
        """
        Render the neon mask effect on detected face.

        Args:
            face_data: Dictionary from face_tracker
            t: Current time for animation
            intensity: Animation intensity (0-1, based on mouth openness)
        """
        landmarks = face_data['landmarks']
        key_points = self.face_tracker.get_key_points(landmarks)

        if not key_points:
            return

        # Get face metrics
        f_width = face_width(landmarks)
        tilt = head_tilt(landmarks)

        # Smooth values for stable animation
        self.smooth_tilt = smooth_value(self.smooth_tilt, tilt, 0.3)

        # Cycle colors slowly
        color_cycle = int(t / 3.0) % len(self.color_names)
        current_color = self.COLORS[self.color_names[color_cycle]]
        next_color = self.COLORS[self.color_names[(color_cycle + 1) % len(self.color_names)]]

        # Interpolate between colors
        blend = (t / 3.0) % 1.0
        blend = 0.5 - 0.5 * math.cos(blend * math.pi)  # Smooth blend
        color = tuple(c1 + (c2 - c1) * blend for c1, c2 in zip(current_color, next_color))

        # Get key positions
        forehead = key_points['forehead']
        nose = key_points['nose_tip']
        left_eye = key_points['left_eye_outer']
        right_eye = key_points['right_eye_outer']

        # === Draw mask elements ===

        # 1. Forehead circles (animated)
        circle_y = forehead[1] - f_width * 0.1
        pulse_offset = math.sin(t * 4.0) * 0.01 * intensity

        # Left decoration
        left_circle = build_circle(
            (forehead[0] - f_width * 0.3, circle_y + pulse_offset),
            f_width * 0.08
        )
        self._draw_vertices(left_circle, GL_LINE_LOOP, color, 0.9, intensity)

        # Right decoration
        right_circle = build_circle(
            (forehead[0] + f_width * 0.3, circle_y + pulse_offset),
            f_width * 0.08
        )
        self._draw_vertices(right_circle, GL_LINE_LOOP, color, 0.9, intensity)

        # 2. Horns (with tilt rotation)
        left_horn = build_horn(
            (forehead[0] - f_width * 0.25, forehead[1] - 0.02),
            scale=f_width * 2.5,
            angle=self.smooth_tilt - 0.3,
            flip=False
        )
        self._draw_vertices(left_horn, GL_LINE_STRIP, color, 0.95, intensity)

        right_horn = build_horn(
            (forehead[0] + f_width * 0.25, forehead[1] - 0.02),
            scale=f_width * 2.5,
            angle=self.smooth_tilt + 0.3,
            flip=True
        )
        self._draw_vertices(right_horn, GL_LINE_STRIP, color, 0.95, intensity)

        # 3. Eye accent lines
        eye_width = abs(right_eye[0] - left_eye[0])

        # Left eye zigzag
        left_start = (left_eye[0] - eye_width * 0.1, left_eye[1])
        left_end = (left_eye[0] - eye_width * 0.4, left_eye[1] - 0.02)
        left_zig = build_zigzag_line(left_start, left_end,
                                     amplitude=0.008 + 0.008 * intensity,
                                     frequency=4)
        self._draw_vertices(left_zig, GL_LINE_STRIP, color, 0.8, intensity * 0.5)

        # Right eye zigzag
        right_start = (right_eye[0] + eye_width * 0.1, right_eye[1])
        right_end = (right_eye[0] + eye_width * 0.4, right_eye[1] - 0.02)
        right_zig = build_zigzag_line(right_start, right_end,
                                      amplitude=0.008 + 0.008 * intensity,
                                      frequency=4)
        self._draw_vertices(right_zig, GL_LINE_STRIP, color, 0.8, intensity * 0.5)

        # 4. Nose bridge line
        bridge_lines = build_neon_lines(
            (forehead[0], forehead[1] + 0.02),
            (nose[0], nose[1] - 0.03),
            num_lines=2,
            spacing=0.008
        )
        for line in bridge_lines:
            self._draw_vertices(line, GL_LINES, color, 0.6 + 0.3 * intensity, intensity)

        # 5. Star decorations (appear when mouth is open)
        if intensity > 0.3:
            star_alpha = (intensity - 0.3) / 0.7
            star_size = f_width * 0.06 * (1 + 0.3 * intensity)

            # Animated star positions
            star_angle = t * 2.0
            star_offset = 0.02 * math.sin(t * 3.0)

            # Left star
            left_star = build_star(
                (left_eye[0] - f_width * 0.2, left_eye[1] - 0.04 + star_offset),
                star_size
            )
            self._draw_vertices(left_star, GL_LINE_LOOP, color, star_alpha * 0.8, 1.0)

            # Right star
            right_star = build_star(
                (right_eye[0] + f_width * 0.2, right_eye[1] - 0.04 - star_offset),
                star_size
            )
            self._draw_vertices(right_star, GL_LINE_LOOP, color, star_alpha * 0.8, 1.0)

        # 6. Cheek accent circles
        left_cheek_pos = key_points.get('left_cheek', (left_eye[0], nose[1]))
        right_cheek_pos = key_points.get('right_cheek', (right_eye[0], nose[1]))

        cheek_pulse = math.sin(t * 5.0) * 0.5 + 0.5
        cheek_radius = f_width * 0.04 * (1 + 0.2 * cheek_pulse * intensity)

        left_cheek_circle = build_circle(
            (left_cheek_pos[0] - f_width * 0.1, left_cheek_pos[1]),
            cheek_radius
        )
        self._draw_vertices(left_cheek_circle, GL_LINE_LOOP, color, 0.5 + 0.3 * intensity, intensity)

        right_cheek_circle = build_circle(
            (right_cheek_pos[0] + f_width * 0.1, right_cheek_pos[1]),
            cheek_radius
        )
        self._draw_vertices(right_cheek_circle, GL_LINE_LOOP, color, 0.5 + 0.3 * intensity, intensity)

    def _update_background(self, frame: np.ndarray):
        """Update background texture with camera frame."""
        # Convert BGR to RGB and flip
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        glBindTexture(GL_TEXTURE_2D, self.bg_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, rgb_frame)

    def _render_background(self):
        """Render camera background using a fullscreen quad."""
        # For simplicity, we'll use OpenCV window for background
        # OpenGL overlay would require additional shader and texture setup
        pass

    def run(self):
        """Main application loop."""
        if not self.initialize():
            return

        fps_time = time.time()
        fps_count = 0
        current_fps = 0

        # For hybrid rendering: we'll use OpenCV window with OpenGL overlay info
        # This is simpler and more compatible

        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()

                # Check for ESC key
                if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                    break

                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)

                # Resize if needed
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                # Track face
                face_data = self.face_tracker.process_frame(frame)

                # Time for animation
                t = time.time() - self.start_time

                # Calculate mouth intensity
                if face_data['detected']:
                    raw_mouth = mouth_openness(face_data['landmarks'])
                    self.smooth_mouth = smooth_value(self.smooth_mouth, raw_mouth, 0.4)
                else:
                    self.smooth_mouth = smooth_value(self.smooth_mouth, 0.0, 0.1)

                # Clear OpenGL
                glClearColor(0.0, 0.0, 0.0, 0.0)
                glClear(GL_COLOR_BUFFER_BIT)

                # Use shader
                glUseProgram(self.shader_program)

                # Set projection matrix
                proj = self._create_ortho_projection()
                glUniformMatrix4fv(self.u_projection, 1, GL_FALSE, proj)
                glUniform2f(self.u_offset, 0.0, 0.0)
                glUniform1f(self.u_scale, 1.0)
                glUniform1f(self.u_time, t)

                # Render mask if face detected
                if face_data['detected']:
                    self._render_neon_mask(face_data, t, self.smooth_mouth)

                # Read OpenGL framebuffer
                glReadBuffer(GL_FRONT)
                pixels = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_UNSIGNED_BYTE)
                gl_image = np.frombuffer(pixels, dtype=np.uint8).reshape(self.height, self.width, 4)
                gl_image = cv2.flip(gl_image, 0)  # Flip vertically

                # Composite: overlay OpenGL on camera frame
                # Extract alpha channel
                alpha = gl_image[:, :, 3:4] / 255.0
                gl_rgb = gl_image[:, :, :3]

                # Convert frame to float
                frame_float = frame.astype(np.float32)
                gl_float = cv2.cvtColor(gl_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)

                # Alpha blend
                composite = frame_float * (1 - alpha) + gl_float * alpha
                composite = composite.astype(np.uint8)

                # Draw HUD info
                self._draw_hud(composite, face_data, current_fps)

                # Show in OpenCV window (more compatible than pure OpenGL)
                cv2.imshow("Neon Mask AR Filter", composite)

                # Swap OpenGL buffers
                glfw.swap_buffers(self.window)

                # Handle OpenCV key (for backup exit)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

                # FPS counter
                fps_count += 1
                if time.time() - fps_time >= 1.0:
                    current_fps = fps_count
                    fps_count = 0
                    fps_time = time.time()

        except KeyboardInterrupt:
            print("\n[AR FILTER] Interrupted by user")

        finally:
            self.cleanup()

    def _draw_hud(self, frame: np.ndarray, face_data: dict, fps: int):
        """Draw heads-up display information."""
        # Background panel
        cv2.rectangle(frame, (10, 10), (300, 100), (20, 20, 30), -1)
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 200, 255), 1)

        # Title
        cv2.putText(frame, "NEON MASK AR FILTER", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        # FPS
        fps_color = (0, 255, 100) if fps >= 25 else (0, 200, 255)
        cv2.putText(frame, f"FPS: {fps}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1, cv2.LINE_AA)

        # Face status
        status = "FACE: DETECTED" if face_data['detected'] else "FACE: SEARCHING..."
        status_color = (0, 255, 100) if face_data['detected'] else (100, 100, 255)
        cv2.putText(frame, status, (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

        # Mouth intensity bar
        if face_data['detected']:
            bar_width = int(150 * self.smooth_mouth)
            cv2.rectangle(frame, (130, 52), (280, 68), (50, 50, 60), -1)
            cv2.rectangle(frame, (130, 52), (130 + bar_width, 68), (0, 255, 255), -1)
            cv2.putText(frame, "INTENSITY", (130, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        # Instructions
        cv2.putText(frame, "Press ESC to exit | Open mouth for more effects",
                    (10, self.height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

    def cleanup(self):
        """Release all resources."""
        print("\n[AR FILTER] Cleaning up...")

        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None

        # Release face tracker
        if self.face_tracker:
            self.face_tracker.release()

        # Cleanup OpenGL
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.bg_texture:
            glDeleteTextures(1, [self.bg_texture])
        if self.shader_program:
            glDeleteProgram(self.shader_program)

        # Cleanup GLFW
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()

        # Cleanup OpenCV
        cv2.destroyAllWindows()

        print("[AR FILTER] Shutdown complete!")


def run_ar_filter():
    """
    Entry point for the AR filter.

    This function can be called from external code to launch the filter.
    """
    if not OPENGL_AVAILABLE:
        print("[AR FILTER ERROR] Required packages not installed!")
        print("Please install: pip install PyOpenGL glfw")
        return

    filter_app = NeonMaskFilter(width=1280, height=720)
    filter_app.run()


# Allow running as standalone
if __name__ == "__main__":
    run_ar_filter()
