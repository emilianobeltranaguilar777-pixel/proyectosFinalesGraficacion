"""Minimal OpenGL + GLFW AR filter app."""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import numpy as np

from .metrics import (
    face_width,
    halo_radius,
    head_position,
    mouth_open_ratio,
    mouth_reference_points,
    smooth_value,
)
from .primitives import create_quad, create_sphere


class ARFilterApp:
    """Runs an isolated OpenGL loop for the AR filter."""

    def __init__(self, width: int = 960, height: int = 720):
        self.width = width
        self.height = height
        self.face_tracker = None
        self.halo_angle = 0.0
        self.halo_speed = 1.0
        self.mouth_scale = 0.1
        self.mouth_color = (0.1, 0.3, 1.0)
        self._quad_vertices = np.array(create_quad(), dtype=np.float32)
        self._sphere_vertices = np.array(create_sphere(radius=0.05, segments=8, rings=8), dtype=np.float32)
        self._gl = None
        self._glfw = None
        self._shader_program = None
        self._vao_quad = None
        self._vao_sphere = None

    def _to_clip_space(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Converts normalized landmark coordinates to clip space."""
        x, y = point
        return (x * 2.0) - 1.0, 1.0 - (y * 2.0)

    def _init_window(self):
        import glfw

        if not glfw.init():
            raise RuntimeError("No se pudo inicializar GLFW")
        self._glfw = glfw
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        window = glfw.create_window(self.width, self.height, "AR Filter", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("No se pudo crear la ventana GLFW")
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        return window

    def _compile_shader(self, vertex_path: str, fragment_path: str):
        from OpenGL import GL

        with open(vertex_path, "r", encoding="utf-8") as f:
            vertex_src = f.read()
        with open(fragment_path, "r", encoding="utf-8") as f:
            fragment_src = f.read()

        vert_shader = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vert_shader, vertex_src)
        GL.glCompileShader(vert_shader)

        frag_shader = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(frag_shader, fragment_src)
        GL.glCompileShader(frag_shader)

        program = GL.glCreateProgram()
        GL.glAttachShader(program, vert_shader)
        GL.glAttachShader(program, frag_shader)
        GL.glLinkProgram(program)

        GL.glDeleteShader(vert_shader)
        GL.glDeleteShader(frag_shader)
        return program

    def _setup_buffers(self):
        from OpenGL import GL

        self._gl = GL
        program = self._compile_shader(
            "ar_filter/shaders/basic.vert", "ar_filter/shaders/basic.frag"
        )
        self._shader_program = program

        vao_quad = GL.glGenVertexArrays(1)
        vbo_quad = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao_quad)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_quad)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._quad_vertices.nbytes, self._quad_vertices, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        vao_sphere = GL.glGenVertexArrays(1)
        vbo_sphere = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao_sphere)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_sphere)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._sphere_vertices.nbytes, self._sphere_vertices, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        self._vao_quad = vao_quad
        self._vao_sphere = vao_sphere

    def _draw_quads(
        self,
        centers: List[Tuple[float, float]],
        width: float,
        color: Tuple[float, float, float],
        scale_y: float,
    ):
        GL = self._gl
        GL.glUseProgram(self._shader_program)
        color_loc = GL.glGetUniformLocation(self._shader_program, "u_color")
        GL.glUniform3f(color_loc, *color)
        GL.glBindVertexArray(self._vao_quad)
        transform_loc = GL.glGetUniformLocation(self._shader_program, "u_transform")
        for cx, cy in centers:
            transform = np.array(
                [
                    [width, 0.0, 0.0, cx],
                    [0.0, scale_y, 0.0, cy],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            GL.glUniformMatrix4fv(transform_loc, 1, GL.GL_FALSE, transform)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
        GL.glBindVertexArray(0)

    def _draw_halo(self, center: Tuple[float, float], radius: float):
        GL = self._gl
        GL.glUseProgram(self._shader_program)
        color_loc = GL.glGetUniformLocation(self._shader_program, "u_color")
        GL.glUniform3f(color_loc, 0.6, 0.9, 1.0)
        transform_loc = GL.glGetUniformLocation(self._shader_program, "u_transform")
        for i in range(10):
            angle = self.halo_angle + (2 * math.pi * i) / 10.0
            x = math.cos(angle) * radius
            z = math.sin(angle) * radius
            translate = np.array(
                [
                    [1.0, 0.0, 0.0, center[0] + x],
                    [0.0, 1.0, 0.0, center[1]],
                    [0.0, 0.0, 1.0, z],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            GL.glUniformMatrix4fv(transform_loc, 1, GL.GL_FALSE, translate)
            GL.glBindVertexArray(self._vao_sphere)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, len(self._sphere_vertices))
        GL.glBindVertexArray(0)

    def run(self):
        import cv2
        from .face_tracker import FaceTracker

        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("No se pudo abrir la cámara para el filtro AR")
            cap.release()
            return

        window = self._init_window()
        glfw = self._glfw

        self._setup_buffers()
        GL = self._gl
        self.face_tracker = FaceTracker()
        last_time = time.perf_counter()
        halo_center = (0.0, 0.1)
        mouth_centers: List[Tuple[float, float]] = [(0.0, -0.2), (0.1, -0.2)]
        quad_width = 0.08

        while not glfw.window_should_close(window):
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            self.halo_angle += self.halo_speed * dt

            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            landmarks = self.face_tracker.process(frame)
            mouth_ratio = 0.0
            halo_r = 0.2
            face_w = 0.2
            if landmarks:
                halo_r = halo_radius(landmarks)
                mouth_ratio = mouth_open_ratio(landmarks)
                head_pos = head_position(landmarks)
                halo_center = self._to_clip_space(head_pos)
                face_w = face_width(landmarks)
                left, right, mouth_y = mouth_reference_points(landmarks)
                quad_width = max(0.04, min(0.2, face_w * 0.12))
                mouth_y_clip = self._to_clip_space((0.0, mouth_y))[1]
                mouth_centers = [
                    (self._to_clip_space(left)[0], mouth_y_clip),
                    (self._to_clip_space(right)[0], mouth_y_clip),
                ]

            target_scale = 0.05 + mouth_ratio * 0.35
            target_color = (0.6, 0.0, 0.7) if mouth_ratio > 0.2 else (0.1, 0.3, 1.0)
            self.mouth_scale = smooth_value(self.mouth_scale, target_scale, min(dt * 6.0, 1.0))
            self.mouth_color = tuple(
                smooth_value(c, t, min(dt * 6.0, 1.0)) for c, t in zip(self.mouth_color, target_color)
            )

            GL.glViewport(0, 0, self.width, self.height)
            GL.glClearColor(0.02, 0.02, 0.04, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self._draw_halo(halo_center, halo_r)
            self._draw_quads(mouth_centers, quad_width, self.mouth_color, self.mouth_scale)

            glfw.swap_buffers(window)
            glfw.poll_events()
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break

        cap.release()
        if self.face_tracker:
            self.face_tracker.close()
        glfw.terminate()


def run_ar_filter():
    """Entry point to run the AR filter standalone."""
    app = ARFilterApp()
    app.run()
