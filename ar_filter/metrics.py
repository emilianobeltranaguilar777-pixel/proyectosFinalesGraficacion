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


def forehead_center(landmarks: List[Tuple[float, float, float]],
                    forehead_idx: int = 10) -> Tuple[float, float, float]:
    """
    Get forehead center position for halo placement.

    Uses landmark 10 (top of forehead) which is more stable for
    positioning elements above the head.

    Args:
        landmarks: List of (x, y, z) normalized landmark coordinates
        forehead_idx: Index of forehead landmark (default 10)

    Returns:
        (x, y, z) position of forehead center
    """
    if not landmarks or len(landmarks) <= forehead_idx:
        return (0.5, 0.3, 0.0)

    return landmarks[forehead_idx]


def mouth_center(landmarks: List[Tuple[float, float, float]],
                 top_lip_idx: int = 13,
                 bottom_lip_idx: int = 14,
                 left_corner_idx: int = 61,
                 right_corner_idx: int = 291) -> Tuple[float, float, float]:
    """
    Calculate the true center of the mouth using multiple landmarks.

    Uses lip center and corners for accurate positioning.

    Args:
        landmarks: List of (x, y, z) normalized landmark coordinates
        top_lip_idx: Index of top lip center
        bottom_lip_idx: Index of bottom lip center
        left_corner_idx: Index of left mouth corner
        right_corner_idx: Index of right mouth corner

    Returns:
        (x, y, z) center position of mouth
    """
    required_max = max(top_lip_idx, bottom_lip_idx, left_corner_idx, right_corner_idx)
    if not landmarks or len(landmarks) <= required_max:
        return (0.5, 0.7, 0.0)

    top = landmarks[top_lip_idx]
    bottom = landmarks[bottom_lip_idx]
    left = landmarks[left_corner_idx]
    right = landmarks[right_corner_idx]

    # Center X is average of corners, Y is average of top/bottom lips
    center_x = (left[0] + right[0]) / 2.0
    center_y = (top[1] + bottom[1]) / 2.0
    center_z = (top[2] + bottom[2]) / 2.0

    return (center_x, center_y, center_z)


def mouth_width(landmarks: List[Tuple[float, float, float]],
                left_corner_idx: int = 61,
                right_corner_idx: int = 291) -> float:
    """
    Calculate mouth width from corner to corner.

    Args:
        landmarks: List of (x, y, z) normalized landmark coordinates
        left_corner_idx: Index of left mouth corner
        right_corner_idx: Index of right mouth corner

    Returns:
        Width of mouth as distance between corners
    """
    if not landmarks or len(landmarks) <= max(left_corner_idx, right_corner_idx):
        return 0.1

    left = landmarks[left_corner_idx]
    right = landmarks[right_corner_idx]

    dx = right[0] - left[0]
    dy = right[1] - left[1]

    return math.sqrt(dx * dx + dy * dy)


def halo_sphere_positions_v2(forehead: Tuple[float, float, float],
                              radius: float,
                              num_spheres: int,
                              rotation_angle: float,
                              height_offset: float = 0.08) -> List[Tuple[float, float, float]]:
    """
    Calculate positions of spheres arranged in a halo above the forehead.

    This version uses the forehead position directly for better placement.

    Args:
        forehead: (x, y, z) position of forehead center
        radius: Radius of the halo ring
        num_spheres: Number of spheres in the halo
        rotation_angle: Current rotation angle in radians
        height_offset: How far above forehead to place the halo

    Returns:
        List of (x, y, z) positions for each sphere
    """
    if num_spheres <= 0:
        return []

    positions = []
    angle_step = (2.0 * math.pi) / num_spheres

    for i in range(num_spheres):
        angle = rotation_angle + (i * angle_step)
        x = forehead[0] + radius * math.cos(angle)
        y = forehead[1] - height_offset  # Above the forehead (Y decreases upward)
        z = forehead[2] + radius * math.sin(angle) * 0.4  # Slight ellipse for depth
        positions.append((x, y, z))

    return positions


# ============================================================================
# Robot Mouth Metrics
# ============================================================================

def robot_mouth_bar_color(openness: float) -> Tuple[float, float, float]:
    """
    Calculate bar color based on mouth openness (blue → yellow → red).

    Args:
        openness: Mouth openness ratio [0.0, 1.0]

    Returns:
        RGB color tuple with values in [0.0, 1.0]
    """
    openness = clamp(openness, 0.0, 1.0)

    # Three-phase gradient: blue → yellow → red
    if openness < 0.5:
        # Blue to Yellow (0.0 - 0.5)
        t = openness * 2.0
        r = lerp(0.2, 1.0, t)
        g = lerp(0.6, 1.0, t)
        b = lerp(1.0, 0.2, t)
    else:
        # Yellow to Red (0.5 - 1.0)
        t = (openness - 0.5) * 2.0
        r = 1.0
        g = lerp(1.0, 0.2, t)
        b = lerp(0.2, 0.1, t)

    return (r, g, b)


def robot_mouth_bar_heights(
    num_bars: int,
    openness: float,
    time: float,
    base_height: float = 0.3,
    max_height: float = 1.0,
    pulse_freq: float = 8.0
) -> List[float]:
    """
    Calculate heights for N bars based on mouth openness with pulse effect.

    Args:
        num_bars: Number of bars
        openness: Mouth openness ratio [0.0, 1.0]
        time: Current time in seconds (for animation)
        base_height: Minimum bar height when mouth closed
        max_height: Maximum bar height when mouth fully open
        pulse_freq: Frequency of pulsing animation

    Returns:
        List of height values for each bar
    """
    heights = []
    openness = clamp(openness, 0.0, 1.0)

    for i in range(num_bars):
        # Base height scales with openness
        height = lerp(base_height, max_height, openness)

        # Add pulse effect when mouth is open
        if openness > 0.1:
            # Phase offset per bar for wave effect
            phase = (i / max(1, num_bars - 1)) * math.pi * 2.0
            pulse = math.sin(time * pulse_freq + phase) * 0.5 + 0.5
            pulse_amplitude = openness * 0.3
            height += pulse * pulse_amplitude

        heights.append(clamp(height, 0.0, max_height))

    return heights


def estimate_face_yaw(landmarks: List[Tuple[float, float, float]],
                      nose_idx: int = 1,
                      left_cheek_idx: int = 234,
                      right_cheek_idx: int = 454) -> float:
    """
    Estimate face yaw rotation (left-right turn) from landmarks.

    Args:
        landmarks: List of (x, y, z) normalized landmark coordinates
        nose_idx: Index of nose tip landmark
        left_cheek_idx: Index of left cheek landmark
        right_cheek_idx: Index of right cheek landmark

    Returns:
        Yaw angle in radians (negative = looking left, positive = looking right)
    """
    max_idx = max(nose_idx, left_cheek_idx, right_cheek_idx)
    if not landmarks or len(landmarks) <= max_idx:
        return 0.0

    nose = landmarks[nose_idx]
    left = landmarks[left_cheek_idx]
    right = landmarks[right_cheek_idx]

    # Face center X
    center_x = (left[0] + right[0]) / 2.0

    # How far nose is from center (normalized)
    face_width = abs(right[0] - left[0])
    if face_width < 0.01:
        return 0.0

    offset = (nose[0] - center_x) / face_width

    # Convert to approximate angle (max ~45 degrees)
    return clamp(offset * math.pi * 0.25, -math.pi / 4, math.pi / 4)


def estimate_face_roll(landmarks: List[Tuple[float, float, float]],
                       left_eye_idx: int = 33,
                       right_eye_idx: int = 263) -> float:
    """
    Estimate face roll rotation (head tilt) from eye landmarks.

    Args:
        landmarks: List of (x, y, z) normalized landmark coordinates
        left_eye_idx: Index of left eye outer corner
        right_eye_idx: Index of right eye outer corner

    Returns:
        Roll angle in radians (negative = tilted left, positive = tilted right)
    """
    max_idx = max(left_eye_idx, right_eye_idx)
    if not landmarks or len(landmarks) <= max_idx:
        return 0.0

    left_eye = landmarks[left_eye_idx]
    right_eye = landmarks[right_eye_idx]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    # Calculate angle from horizontal
    if abs(dx) < 0.001:
        return 0.0

    return math.atan2(dy, dx)


def mouth_plate_dimensions(
    landmarks: List[Tuple[float, float, float]],
    width_scale: float = 1.2,
    height_ratio: float = 0.45
) -> Tuple[float, float]:
    """
    Calculate robot mouth plate dimensions based on face landmarks.

    Args:
        landmarks: List of (x, y, z) normalized landmark coordinates
        width_scale: Multiplier for plate width relative to mouth width
        height_ratio: Plate height as ratio of width

    Returns:
        Tuple of (plate_width, plate_height)
    """
    m_width = mouth_width(landmarks)
    plate_width = m_width * width_scale
    plate_height = plate_width * height_ratio

    return (plate_width, plate_height)


# ============================================================================
# Orbiting Cubes Metrics
# ============================================================================

def cube_orbit_positions(
    center: Tuple[float, float, float],
    base_radius: float,
    num_cubes: int,
    time: float,
    cube_data: List[dict]
) -> List[Tuple[Tuple[float, float, float], float, float, float]]:
    """
    Calculate positions and rotations for orbiting cubes.

    Args:
        center: (x, y, z) orbit center (face center)
        base_radius: Base radius for orbits (scaled by face size)
        num_cubes: Number of cubes
        time: Current time in seconds
        cube_data: List of dicts with orbit_radius_factor, orbit_speed,
                   spin_speed, wobble_speed, wobble_amp per cube

    Returns:
        List of (position, spin_angle, wobble_x, wobble_z) per cube
    """
    results = []

    for i, data in enumerate(cube_data[:num_cubes]):
        # Individual orbit parameters
        orbit_r = base_radius * data.get('orbit_radius_factor', 1.0)
        orbit_speed = data.get('orbit_speed', 0.8)
        spin_speed = data.get('spin_speed', 1.5)
        wobble_speed = data.get('wobble_speed', 2.0)
        wobble_amp = data.get('wobble_amp', 0.5)

        # Orbital position
        angle = time * orbit_speed + (i * 2.0 * math.pi / num_cubes)
        ox = center[0] + orbit_r * math.cos(angle)
        oz = center[2] + orbit_r * math.sin(angle) * 0.4  # Elliptical
        oy = center[1] + math.sin(time * wobble_speed + i) * wobble_amp * base_radius

        # Rotation angles
        spin_angle = time * spin_speed
        wobble_x = math.sin(time * wobble_speed + i) * 0.4
        wobble_z = math.sin(time * wobble_speed * 1.3 + i) * 0.4

        results.append(((ox, oy, oz), spin_angle, wobble_x, wobble_z))

    return results
