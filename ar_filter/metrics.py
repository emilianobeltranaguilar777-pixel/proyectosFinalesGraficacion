"""
Pure math functions for AR filter metrics.

This module contains ONLY pure functions with no OpenGL dependencies.
All functions are fully testable without camera or graphics context.
"""

import math
from typing import Tuple, List, Optional


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max bounds.

    Args:
        value: The value to clamp
        min_val: Minimum bound
        max_val: Maximum bound

    Returns:
        Clamped value within [min_val, max_val]
    """
    return max(min_val, min(max_val, value))


def face_width(landmarks: List[Tuple[float, float, float]],
               left_idx: int = 234,
               right_idx: int = 454) -> float:
    """
    Calculate face width from landmarks.

    Uses cheek landmarks (234 = left cheek, 454 = right cheek) by default.

    Args:
        landmarks: List of (x, y, z) normalized landmark coordinates
        left_idx: Index of left face boundary landmark
        right_idx: Index of right face boundary landmark

    Returns:
        Face width as Euclidean distance between left and right points
    """
    if not landmarks or len(landmarks) <= max(left_idx, right_idx):
        return 0.0

    left = landmarks[left_idx]
    right = landmarks[right_idx]

    dx = right[0] - left[0]
    dy = right[1] - left[1]

    return math.sqrt(dx * dx + dy * dy)


def halo_radius(face_w: float, scale_factor: float = 1.5) -> float:
    """
    Calculate halo radius based on face width.

    Args:
        face_w: Face width (from face_width function)
        scale_factor: Multiplier for halo size relative to face

    Returns:
        Halo radius (clamped to reasonable range)
    """
    radius = face_w * scale_factor
    return clamp(radius, 0.05, 0.5)


def mouth_openness(landmarks: List[Tuple[float, float, float]],
                   top_lip_idx: int = 13,
                   bottom_lip_idx: int = 14) -> float:
    """
    Calculate how open the mouth is (0.0 = closed, 1.0 = fully open).

    Uses lip landmarks (13 = top lip center, 14 = bottom lip center).

    Args:
        landmarks: List of (x, y, z) normalized landmark coordinates
        top_lip_idx: Index of top lip center landmark
        bottom_lip_idx: Index of bottom lip center landmark

    Returns:
        Mouth openness ratio normalized to [0.0, 1.0]
    """
    if not landmarks or len(landmarks) <= max(top_lip_idx, bottom_lip_idx):
        return 0.0

    top = landmarks[top_lip_idx]
    bottom = landmarks[bottom_lip_idx]

    # Vertical distance between lips
    lip_distance = abs(bottom[1] - top[1])

    # Normalize based on typical mouth opening range
    # Closed mouth: ~0.01-0.02, Open mouth: ~0.05-0.08
    MIN_DIST = 0.01
    MAX_DIST = 0.06

    normalized = (lip_distance - MIN_DIST) / (MAX_DIST - MIN_DIST)
    return clamp(normalized, 0.0, 1.0)


def face_center(landmarks: List[Tuple[float, float, float]],
                nose_idx: int = 1) -> Tuple[float, float, float]:
    """
    Get face center position (using nose tip as reference).

    Args:
        landmarks: List of (x, y, z) normalized landmark coordinates
        nose_idx: Index of nose tip landmark

    Returns:
        (x, y, z) position of face center
    """
    if not landmarks or len(landmarks) <= nose_idx:
        return (0.5, 0.5, 0.0)

    return landmarks[nose_idx]


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between two values.

    Args:
        a: Start value
        b: End value
        t: Interpolation factor [0.0, 1.0]

    Returns:
        Interpolated value
    """
    t = clamp(t, 0.0, 1.0)
    return a + (b - a) * t


def smooth_value(current: float, target: float, smoothing: float = 0.3) -> float:
    """
    Smooth a value towards a target using exponential smoothing.

    Args:
        current: Current value
        target: Target value to move towards
        smoothing: Smoothing factor (0.0 = no smoothing, 1.0 = instant)

    Returns:
        Smoothed value closer to target
    """
    smoothing = clamp(smoothing, 0.0, 1.0)
    return lerp(current, target, smoothing)


def halo_sphere_positions(center: Tuple[float, float, float],
                          radius: float,
                          num_spheres: int,
                          rotation_angle: float) -> List[Tuple[float, float, float]]:
    """
    Calculate positions of spheres arranged in a halo around a center point.

    Args:
        center: (x, y, z) center of the halo
        radius: Radius of the halo ring
        num_spheres: Number of spheres in the halo
        rotation_angle: Current rotation angle in radians

    Returns:
        List of (x, y, z) positions for each sphere
    """
    if num_spheres <= 0:
        return []

    positions = []
    angle_step = (2.0 * math.pi) / num_spheres

    for i in range(num_spheres):
        angle = rotation_angle + (i * angle_step)
        x = center[0] + radius * math.cos(angle)
        y = center[1] - radius * 0.3  # Offset above head
        z = center[2] + radius * math.sin(angle) * 0.5  # Elliptical in Z
        positions.append((x, y, z))

    return positions


def mouth_rect_scale(openness: float,
                     min_scale: float = 0.5,
                     max_scale: float = 2.0) -> float:
    """
    Calculate rectangle scale factor based on mouth openness.

    Args:
        openness: Mouth openness ratio [0.0, 1.0]
        min_scale: Scale when mouth is closed
        max_scale: Scale when mouth is fully open

    Returns:
        Scale factor for mouth rectangles
    """
    return lerp(min_scale, max_scale, openness)


def mouth_rect_color(openness: float,
                     closed_color: Tuple[float, float, float] = (0.2, 0.6, 1.0),
                     open_color: Tuple[float, float, float] = (1.0, 0.3, 0.5)) -> Tuple[float, float, float]:
    """
    Calculate rectangle color based on mouth openness.

    Args:
        openness: Mouth openness ratio [0.0, 1.0]
        closed_color: RGB color when mouth is closed
        open_color: RGB color when mouth is fully open

    Returns:
        Interpolated RGB color tuple
    """
    r = lerp(closed_color[0], open_color[0], openness)
    g = lerp(closed_color[1], open_color[1], openness)
    b = lerp(closed_color[2], open_color[2], openness)
    return (r, g, b)
