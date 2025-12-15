"""
Primitives Module - Geometry generators for AR filter

This module generates vertex data for geometric primitives.
It does NOT draw anything - only returns vertex arrays.

NO OpenGL calls here.
"""

import math
from typing import List, Tuple
import numpy as np


def build_circle(center: Tuple[float, float], radius: float,
                 segments: int = 32) -> np.ndarray:
    """
    Generate vertices for a circle outline.

    Args:
        center: (x, y) center position
        radius: Circle radius
        segments: Number of line segments

    Returns:
        numpy array of vertices [(x, y), ...] for LINE_LOOP
    """
    vertices = []
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))

    return np.array(vertices, dtype=np.float32)


def build_filled_circle(center: Tuple[float, float], radius: float,
                        segments: int = 32) -> np.ndarray:
    """
    Generate vertices for a filled circle (triangle fan).

    Args:
        center: (x, y) center position
        radius: Circle radius
        segments: Number of triangles

    Returns:
        numpy array of vertices for TRIANGLE_FAN
    """
    vertices = [center]  # Center vertex
    for i in range(segments + 1):
        angle = 2.0 * math.pi * i / segments
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))

    return np.array(vertices, dtype=np.float32)


def build_horn(base_pos: Tuple[float, float], scale: float,
               angle: float, flip: bool = False) -> np.ndarray:
    """
    Generate vertices for a curved horn shape.

    Args:
        base_pos: (x, y) base position of the horn
        scale: Scale factor for the horn
        angle: Rotation angle in radians
        flip: If True, mirror horizontally

    Returns:
        numpy array of vertices for LINE_STRIP
    """
    # Horn curve points (relative to base)
    horn_points = [
        (0.0, 0.0),
        (0.02, -0.03),
        (0.05, -0.06),
        (0.07, -0.10),
        (0.08, -0.14),
        (0.07, -0.18),
        (0.05, -0.21),
        (0.02, -0.23),
    ]

    vertices = []
    flip_mult = -1.0 if flip else 1.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    for px, py in horn_points:
        # Apply flip
        px *= flip_mult

        # Apply scale
        px *= scale
        py *= scale

        # Apply rotation
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a

        # Translate to base position
        vertices.append((base_pos[0] + rx, base_pos[1] + ry))

    return np.array(vertices, dtype=np.float32)


def build_mask_outline(face_points: dict, scale: float = 1.0) -> np.ndarray:
    """
    Generate vertices for a mask outline around the face.

    Args:
        face_points: Dictionary with key facial points
                    (nose_tip, left_eye_outer, right_eye_outer, etc.)
        scale: Scale factor

    Returns:
        numpy array of vertices for LINE_LOOP
    """
    if not face_points:
        return np.array([], dtype=np.float32)

    # Build mask around eyes and forehead
    vertices = []

    required_keys = ['left_eyebrow', 'right_eyebrow', 'left_eye_outer',
                     'right_eye_outer', 'nose_tip', 'forehead']

    if not all(k in face_points for k in required_keys):
        return np.array([], dtype=np.float32)

    # Get key points
    forehead = face_points['forehead']
    left_brow = face_points['left_eyebrow']
    right_brow = face_points['right_eyebrow']
    left_eye = face_points['left_eye_outer']
    right_eye = face_points['right_eye_outer']
    nose = face_points['nose_tip']

    # Create smooth mask path
    center_x = (left_eye[0] + right_eye[0]) / 2
    center_y = forehead[1]

    # Top arc
    for i in range(11):
        t = i / 10.0
        angle = math.pi + t * math.pi  # Top half
        radius_x = abs(right_brow[0] - left_brow[0]) * 0.6 * scale
        radius_y = abs(nose[1] - forehead[1]) * 0.3 * scale
        x = center_x + radius_x * math.cos(angle)
        y = center_y + radius_y * math.sin(angle) * 0.5
        vertices.append((x, y))

    # Right side down
    vertices.append((right_brow[0] * scale + (1-scale) * center_x, right_brow[1]))
    vertices.append((right_eye[0] * scale + (1-scale) * center_x, right_eye[1]))

    # Bottom to nose
    vertices.append((nose[0], nose[1] - 0.02))

    # Left side up
    vertices.append((left_eye[0] * scale + (1-scale) * center_x, left_eye[1]))
    vertices.append((left_brow[0] * scale + (1-scale) * center_x, left_brow[1]))

    return np.array(vertices, dtype=np.float32)


def build_cat_ears(forehead_pos: Tuple[float, float],
                   face_width: float, tilt: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate vertices for cat ear shapes.

    Args:
        forehead_pos: (x, y) position of forehead center
        face_width: Width of the face for scaling
        tilt: Head tilt angle in radians

    Returns:
        Tuple of (left_ear_vertices, right_ear_vertices) for LINE_LOOP
    """
    ear_height = face_width * 0.5
    ear_width = face_width * 0.25

    # Base ear shape (triangle with curved base)
    def make_ear(center_x: float, flip: bool) -> np.ndarray:
        vertices = []
        flip_mult = -1.0 if flip else 1.0

        # Ear outline
        base_y = forehead_pos[1] - 0.02
        tip_y = base_y - ear_height

        # Left edge of ear
        vertices.append((center_x - ear_width * 0.5 * flip_mult, base_y))

        # Tip with slight curve
        for i in range(5):
            t = i / 4.0
            x = center_x + (ear_width * 0.1 * flip_mult) * (1 - abs(t - 0.5) * 2)
            y = base_y - ear_height * (0.3 + 0.7 * (1 - (t - 0.5)**2 * 4))
            vertices.append((x, y))

        # Right edge of ear
        vertices.append((center_x + ear_width * 0.5 * flip_mult, base_y))

        return np.array(vertices, dtype=np.float32)

    # Position ears
    ear_offset = face_width * 0.35

    cos_t = math.cos(tilt)
    sin_t = math.sin(tilt)

    # Apply tilt rotation to ear positions
    left_x = forehead_pos[0] - ear_offset
    right_x = forehead_pos[0] + ear_offset

    left_ear = make_ear(left_x, flip=False)
    right_ear = make_ear(right_x, flip=True)

    # Apply tilt rotation
    def rotate_vertices(verts: np.ndarray, center: Tuple[float, float], angle: float) -> np.ndarray:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotated = []
        for v in verts:
            dx = v[0] - center[0]
            dy = v[1] - center[1]
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            rotated.append((center[0] + rx, center[1] + ry))
        return np.array(rotated, dtype=np.float32)

    left_ear = rotate_vertices(left_ear, forehead_pos, tilt)
    right_ear = rotate_vertices(right_ear, forehead_pos, tilt)

    return left_ear, right_ear


def build_halo(center: Tuple[float, float], radius: float,
               thickness: float = 0.02, segments: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate vertices for a halo ring (two circles).

    Args:
        center: (x, y) center position
        radius: Outer radius
        thickness: Ring thickness
        segments: Number of segments per circle

    Returns:
        Tuple of (outer_circle, inner_circle) vertices
    """
    outer = build_circle(center, radius, segments)
    inner = build_circle(center, radius - thickness, segments)

    return outer, inner


def build_neon_lines(start: Tuple[float, float], end: Tuple[float, float],
                     num_lines: int = 3, spacing: float = 0.01) -> List[np.ndarray]:
    """
    Generate parallel neon lines for glow effect.

    Args:
        start: (x, y) start position
        end: (x, y) end position
        num_lines: Number of parallel lines
        spacing: Spacing between lines

    Returns:
        List of line vertex arrays
    """
    lines = []

    # Calculate perpendicular direction
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < 0.001:
        return lines

    # Perpendicular unit vector
    px = -dy / length
    py = dx / length

    # Generate parallel lines
    for i in range(num_lines):
        offset = (i - (num_lines - 1) / 2) * spacing
        ox = px * offset
        oy = py * offset

        line = np.array([
            (start[0] + ox, start[1] + oy),
            (end[0] + ox, end[1] + oy)
        ], dtype=np.float32)
        lines.append(line)

    return lines


def build_zigzag_line(start: Tuple[float, float], end: Tuple[float, float],
                      amplitude: float = 0.02, frequency: int = 8) -> np.ndarray:
    """
    Generate vertices for a zigzag/lightning line.

    Args:
        start: (x, y) start position
        end: (x, y) end position
        amplitude: Zigzag amplitude
        frequency: Number of zigzags

    Returns:
        numpy array of vertices for LINE_STRIP
    """
    vertices = [start]

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < 0.001:
        return np.array([start, end], dtype=np.float32)

    # Unit vectors along and perpendicular to line
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux

    for i in range(1, frequency):
        t = i / frequency
        # Position along line
        x = start[0] + dx * t
        y = start[1] + dy * t
        # Add zigzag offset
        sign = 1 if i % 2 == 0 else -1
        x += px * amplitude * sign
        y += py * amplitude * sign
        vertices.append((x, y))

    vertices.append(end)

    return np.array(vertices, dtype=np.float32)


def build_star(center: Tuple[float, float], outer_radius: float,
               inner_radius: float = None, points: int = 5) -> np.ndarray:
    """
    Generate vertices for a star shape.

    Args:
        center: (x, y) center position
        outer_radius: Radius to outer points
        inner_radius: Radius to inner points (default: outer_radius * 0.4)
        points: Number of star points

    Returns:
        numpy array of vertices for LINE_LOOP
    """
    if inner_radius is None:
        inner_radius = outer_radius * 0.4

    vertices = []
    angle_step = math.pi / points

    for i in range(points * 2):
        angle = -math.pi / 2 + i * angle_step
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        vertices.append((x, y))

    return np.array(vertices, dtype=np.float32)


def build_heart(center: Tuple[float, float], size: float) -> np.ndarray:
    """
    Generate vertices for a heart shape.

    Args:
        center: (x, y) center position
        size: Size scale factor

    Returns:
        numpy array of vertices for LINE_LOOP
    """
    vertices = []
    segments = 40

    for i in range(segments):
        t = 2.0 * math.pi * i / segments

        # Heart parametric equation
        x = 16 * (math.sin(t) ** 3)
        y = 13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)

        # Scale and translate
        x = center[0] + x * size * 0.01
        y = center[1] - y * size * 0.01  # Flip Y

        vertices.append((x, y))

    return np.array(vertices, dtype=np.float32)
