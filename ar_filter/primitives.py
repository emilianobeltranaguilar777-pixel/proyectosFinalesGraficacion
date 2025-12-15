"""
Primitives Module - Geometry generators for AR filter

This module generates vertex data for geometric primitives.
It does NOT draw anything - only returns vertex arrays.

NO OpenGL calls here.

IMPORTANT: All geometry is triangle-based for OpenGL Core Profile compatibility.
           No GL_LINE_* modes are used - lines are rendered as thin quads.
"""

import math
from typing import List, Tuple
import numpy as np


# =============================================================================
# THICK LINE PRIMITIVES (Core Profile Safe - Triangle Based)
# =============================================================================

def build_thick_line(p1: Tuple[float, float], p2: Tuple[float, float],
                     thickness: float = 0.003) -> np.ndarray:
    """
    Generate vertices for a thick line as a quad (2 triangles).

    This is Core Profile safe - uses triangles instead of GL_LINES.

    Args:
        p1: (x, y) start position
        p2: (x, y) end position
        thickness: Line thickness (half-width)

    Returns:
        numpy array of 6 vertices for GL_TRIANGLES (2 triangles)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < 0.0001:
        # Degenerate line - return empty
        return np.array([], dtype=np.float32).reshape(0, 2)

    # Perpendicular unit vector
    px = -dy / length * thickness
    py = dx / length * thickness

    # Four corners of the quad
    v0 = (p1[0] + px, p1[1] + py)  # Start top
    v1 = (p1[0] - px, p1[1] - py)  # Start bottom
    v2 = (p2[0] + px, p2[1] + py)  # End top
    v3 = (p2[0] - px, p2[1] - py)  # End bottom

    # Two triangles: (v0, v1, v2) and (v1, v3, v2)
    vertices = [v0, v1, v2, v1, v3, v2]

    return np.array(vertices, dtype=np.float32)


def build_thick_line_strip(points: List[Tuple[float, float]],
                           thickness: float = 0.003) -> np.ndarray:
    """
    Generate vertices for a connected line strip as triangles.

    Args:
        points: List of (x, y) positions
        thickness: Line thickness

    Returns:
        numpy array of vertices for GL_TRIANGULAR_STRIP
    """
    if len(points) < 2:
        return np.array([], dtype=np.float32).reshape(0, 2)

    vertices = []

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length < 0.0001:
            continue

        # Perpendicular
        px = -dy / length * thickness
        py = dx / length * thickness

        # Add quad for this segment
        v0 = (p1[0] + px, p1[1] + py)
        v1 = (p1[0] - px, p1[1] - py)
        v2 = (p2[0] + px, p2[1] + py)
        v3 = (p2[0] - px, p2[1] - py)

        vertices.extend([v0, v1, v2, v1, v3, v2])

    if not vertices:
        return np.array([], dtype=np.float32).reshape(0, 2)

    return np.array(vertices, dtype=np.float32)


def build_thick_circle(center: Tuple[float, float], radius: float,
                       thickness: float = 0.003, segments: int = 32) -> np.ndarray:
    """
    Generate vertices for a thick circle outline as triangles.

    Args:
        center: (x, y) center position
        radius: Circle radius
        thickness: Line thickness
        segments: Number of segments

    Returns:
        numpy array of vertices for GL_TRIANGLES
    """
    vertices = []

    for i in range(segments):
        angle1 = 2.0 * math.pi * i / segments
        angle2 = 2.0 * math.pi * (i + 1) / segments

        # Inner and outer points for segment
        inner1 = (center[0] + (radius - thickness) * math.cos(angle1),
                  center[1] + (radius - thickness) * math.sin(angle1))
        outer1 = (center[0] + (radius + thickness) * math.cos(angle1),
                  center[1] + (radius + thickness) * math.sin(angle1))
        inner2 = (center[0] + (radius - thickness) * math.cos(angle2),
                  center[1] + (radius - thickness) * math.sin(angle2))
        outer2 = (center[0] + (radius + thickness) * math.cos(angle2),
                  center[1] + (radius + thickness) * math.sin(angle2))

        # Two triangles per segment
        vertices.extend([outer1, inner1, outer2, inner1, inner2, outer2])

    return np.array(vertices, dtype=np.float32)


def build_thick_star(center: Tuple[float, float], outer_radius: float,
                     inner_radius: float = None, points: int = 5,
                     thickness: float = 0.003) -> np.ndarray:
    """
    Generate vertices for a thick star outline as triangles.

    Args:
        center: (x, y) center position
        outer_radius: Radius to outer points
        inner_radius: Radius to inner points
        points: Number of star points
        thickness: Line thickness

    Returns:
        numpy array of vertices for GL_TRIANGLES
    """
    if inner_radius is None:
        inner_radius = outer_radius * 0.4

    # Generate star points
    star_points = []
    angle_step = math.pi / points

    for i in range(points * 2):
        angle = -math.pi / 2 + i * angle_step
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        star_points.append((x, y))

    # Close the loop
    star_points.append(star_points[0])

    return build_thick_line_strip(star_points, thickness)


def build_thick_polygon(points: List[Tuple[float, float]],
                        thickness: float = 0.003, closed: bool = True) -> np.ndarray:
    """
    Generate vertices for a thick polygon outline as triangles.

    Args:
        points: List of (x, y) vertices
        thickness: Line thickness
        closed: If True, connect last point to first

    Returns:
        numpy array of vertices for GL_TRIANGLES
    """
    if len(points) < 2:
        return np.array([], dtype=np.float32).reshape(0, 2)

    all_points = list(points)
    if closed and len(points) >= 3:
        all_points.append(points[0])

    return build_thick_line_strip(all_points, thickness)


# =============================================================================
# FILLED PRIMITIVES (Already Core Profile Safe)
# =============================================================================

def build_filled_quad(p1: Tuple[float, float], p2: Tuple[float, float],
                      p3: Tuple[float, float], p4: Tuple[float, float]) -> np.ndarray:
    """
    Generate vertices for a filled quad as 2 triangles.

    Args:
        p1, p2, p3, p4: Four corners in order

    Returns:
        numpy array of 6 vertices for GL_TRIANGLES
    """
    return np.array([p1, p2, p3, p2, p4, p3], dtype=np.float32)


# =============================================================================
# LEGACY FUNCTIONS (Now return thick geometry)
# =============================================================================

def build_circle(center: Tuple[float, float], radius: float,
                 segments: int = 32, thickness: float = 0.003) -> np.ndarray:
    """
    Generate vertices for a circle outline as triangles.

    NOTE: Returns triangle-based geometry for Core Profile compatibility.

    Args:
        center: (x, y) center position
        radius: Circle radius
        segments: Number of segments
        thickness: Line thickness

    Returns:
        numpy array of vertices for GL_TRIANGLES
    """
    return build_thick_circle(center, radius, thickness, segments)


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
               angle: float, flip: bool = False,
               thickness: float = 0.004) -> np.ndarray:
    """
    Generate vertices for a curved horn shape as triangles.

    NOTE: Returns triangle-based geometry for Core Profile compatibility.

    Args:
        base_pos: (x, y) base position of the horn
        scale: Scale factor for the horn
        angle: Rotation angle in radians
        flip: If True, mirror horizontally
        thickness: Line thickness

    Returns:
        numpy array of vertices for GL_TRIANGLES
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

    # Convert to thick line strip (triangles)
    return build_thick_line_strip(vertices, thickness)


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
                     num_lines: int = 3, spacing: float = 0.01,
                     thickness: float = 0.002) -> List[np.ndarray]:
    """
    Generate parallel neon lines for glow effect as triangles.

    NOTE: Returns triangle-based geometry for Core Profile compatibility.

    Args:
        start: (x, y) start position
        end: (x, y) end position
        num_lines: Number of parallel lines
        spacing: Spacing between lines
        thickness: Line thickness

    Returns:
        List of triangle vertex arrays for GL_TRIANGLES
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

    # Generate parallel lines as thick quads
    for i in range(num_lines):
        offset = (i - (num_lines - 1) / 2) * spacing
        ox = px * offset
        oy = py * offset

        p1 = (start[0] + ox, start[1] + oy)
        p2 = (end[0] + ox, end[1] + oy)

        line = build_thick_line(p1, p2, thickness)
        if len(line) > 0:
            lines.append(line)

    return lines


def build_zigzag_line(start: Tuple[float, float], end: Tuple[float, float],
                      amplitude: float = 0.02, frequency: int = 8,
                      thickness: float = 0.003) -> np.ndarray:
    """
    Generate vertices for a zigzag/lightning line as triangles.

    NOTE: Returns triangle-based geometry for Core Profile compatibility.

    Args:
        start: (x, y) start position
        end: (x, y) end position
        amplitude: Zigzag amplitude
        frequency: Number of zigzags
        thickness: Line thickness

    Returns:
        numpy array of vertices for GL_TRIANGLES
    """
    vertices = [start]

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)

    if length < 0.001:
        return build_thick_line(start, end, thickness)

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

    # Convert to thick line strip
    return build_thick_line_strip(vertices, thickness)


def build_star(center: Tuple[float, float], outer_radius: float,
               inner_radius: float = None, points: int = 5,
               thickness: float = 0.003) -> np.ndarray:
    """
    Generate vertices for a star shape as triangles.

    NOTE: Returns triangle-based geometry for Core Profile compatibility.

    Args:
        center: (x, y) center position
        outer_radius: Radius to outer points
        inner_radius: Radius to inner points (default: outer_radius * 0.4)
        points: Number of star points
        thickness: Line thickness

    Returns:
        numpy array of vertices for GL_TRIANGLES
    """
    return build_thick_star(center, outer_radius, inner_radius, points, thickness)


def build_heart(center: Tuple[float, float], size: float,
                thickness: float = 0.003) -> np.ndarray:
    """
    Generate vertices for a heart shape as triangles.

    NOTE: Returns triangle-based geometry for Core Profile compatibility.

    Args:
        center: (x, y) center position
        size: Size scale factor
        thickness: Line thickness

    Returns:
        numpy array of vertices for GL_TRIANGLES
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

    # Close the loop and convert to thick polygon
    vertices.append(vertices[0])
    return build_thick_line_strip(vertices, thickness)
