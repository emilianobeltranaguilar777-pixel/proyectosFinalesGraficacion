"""
Metrics Module - Pure mathematical functions for face analysis

This module provides ONLY mathematical calculations.
NO OpenGL, NO MediaPipe, NO rendering.

All functions are pure and testable.
"""

import math
from typing import List, Tuple, Optional


def face_width(landmarks: List[Tuple[float, float, float]]) -> float:
    """
    Calculate the normalized width of the face.

    Uses the distance between left and right cheek landmarks.

    Args:
        landmarks: List of (x, y, z) normalized coordinates

    Returns:
        Normalized face width (0-1 range)
    """
    if not landmarks or len(landmarks) < 455:
        return 0.0

    # Left cheek (234) to right cheek (454)
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]

    dx = right_cheek[0] - left_cheek[0]
    dy = right_cheek[1] - left_cheek[1]

    return math.sqrt(dx * dx + dy * dy)


def face_height(landmarks: List[Tuple[float, float, float]]) -> float:
    """
    Calculate the normalized height of the face.

    Uses the distance from forehead to chin.

    Args:
        landmarks: List of (x, y, z) normalized coordinates

    Returns:
        Normalized face height (0-1 range)
    """
    if not landmarks or len(landmarks) < 153:
        return 0.0

    # Forehead (10) to chin (152)
    forehead = landmarks[10]
    chin = landmarks[152]

    dx = chin[0] - forehead[0]
    dy = chin[1] - forehead[1]

    return math.sqrt(dx * dx + dy * dy)


def mouth_openness(landmarks: List[Tuple[float, float, float]]) -> float:
    """
    Calculate how open the mouth is (0 = closed, 1 = fully open).

    Uses the vertical distance between top and bottom lip,
    normalized by face height.

    Args:
        landmarks: List of (x, y, z) normalized coordinates

    Returns:
        Mouth openness ratio (0-1, clamped)
    """
    if not landmarks or len(landmarks) < 153:
        return 0.0

    # Top lip (13) and bottom lip (14)
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]

    # Vertical distance between lips
    lip_distance = abs(bottom_lip[1] - top_lip[1])

    # Normalize by face height
    f_height = face_height(landmarks)
    if f_height < 0.01:  # Avoid division by zero
        return 0.0

    # Typical open mouth is about 15-20% of face height
    openness = lip_distance / f_height

    # Scale to 0-1 range (0.02 = closed, 0.15 = fully open)
    normalized = (openness - 0.02) / 0.13

    return max(0.0, min(1.0, normalized))


def head_tilt(landmarks: List[Tuple[float, float, float]]) -> float:
    """
    Calculate head tilt angle in radians.

    Positive = tilted right, Negative = tilted left.

    Args:
        landmarks: List of (x, y, z) normalized coordinates

    Returns:
        Tilt angle in radians
    """
    if not landmarks or len(landmarks) < 455:
        return 0.0

    # Use eye corners to determine tilt
    left_eye = landmarks[33]   # Left eye outer corner
    right_eye = landmarks[263]  # Right eye outer corner

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    # Calculate angle from horizontal
    if abs(dx) < 0.001:  # Avoid division by zero
        return 0.0

    return math.atan2(dy, dx)


def head_rotation_y(landmarks: List[Tuple[float, float, float]]) -> float:
    """
    Estimate head rotation around Y axis (left-right turn).

    Uses the asymmetry between left and right eye distances to nose.
    Positive = turned right, Negative = turned left.

    Args:
        landmarks: List of (x, y, z) normalized coordinates

    Returns:
        Estimated Y rotation in radians (-pi/4 to pi/4 range)
    """
    if not landmarks or len(landmarks) < 455:
        return 0.0

    nose = landmarks[1]  # Nose tip
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # Distance from nose to each eye
    dist_left = math.sqrt((nose[0] - left_eye[0])**2 + (nose[1] - left_eye[1])**2)
    dist_right = math.sqrt((nose[0] - right_eye[0])**2 + (nose[1] - right_eye[1])**2)

    total = dist_left + dist_right
    if total < 0.01:
        return 0.0

    # Asymmetry ratio
    asymmetry = (dist_right - dist_left) / total

    # Scale to reasonable rotation range
    return asymmetry * (math.pi / 4)


def face_center(landmarks: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """
    Calculate the center of the face.

    Args:
        landmarks: List of (x, y, z) normalized coordinates

    Returns:
        (x, y) center coordinates (normalized)
    """
    if not landmarks or len(landmarks) < 2:
        return (0.5, 0.5)

    # Use nose tip as approximate center
    nose = landmarks[1]
    return (nose[0], nose[1])


def eye_center(landmarks: List[Tuple[float, float, float]],
               left: bool = True) -> Tuple[float, float]:
    """
    Calculate the center of an eye.

    Args:
        landmarks: List of (x, y, z) normalized coordinates
        left: If True, return left eye center; else right eye

    Returns:
        (x, y) eye center coordinates (normalized)
    """
    if not landmarks or len(landmarks) < 455:
        return (0.5, 0.5)

    if left:
        outer = landmarks[33]
        inner = landmarks[133]
    else:
        outer = landmarks[263]
        inner = landmarks[362]

    return ((outer[0] + inner[0]) / 2, (outer[1] + inner[1]) / 2)


def eyebrow_raise(landmarks: List[Tuple[float, float, float]]) -> float:
    """
    Detect eyebrow raise amount (0 = normal, 1 = raised).

    Args:
        landmarks: List of (x, y, z) normalized coordinates

    Returns:
        Eyebrow raise ratio (0-1)
    """
    if not landmarks or len(landmarks) < 455:
        return 0.0

    # Left eyebrow (70) and left eye (33)
    left_brow = landmarks[70]
    left_eye = landmarks[33]

    # Right eyebrow (300) and right eye (263)
    right_brow = landmarks[300]
    right_eye = landmarks[263]

    # Average vertical distance from eyebrows to eyes
    left_dist = left_eye[1] - left_brow[1]
    right_dist = right_eye[1] - right_brow[1]
    avg_dist = (left_dist + right_dist) / 2

    # Normalize by face height
    f_height = face_height(landmarks)
    if f_height < 0.01:
        return 0.0

    normalized = avg_dist / f_height

    # Scale: 0.08 = normal, 0.12 = raised
    raise_amount = (normalized - 0.08) / 0.04

    return max(0.0, min(1.0, raise_amount))


def smile_intensity(landmarks: List[Tuple[float, float, float]]) -> float:
    """
    Detect smile intensity (0 = neutral, 1 = big smile).

    Uses mouth width relative to face width.

    Args:
        landmarks: List of (x, y, z) normalized coordinates

    Returns:
        Smile intensity ratio (0-1)
    """
    if not landmarks or len(landmarks) < 455:
        return 0.0

    # Mouth corners
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]

    # Mouth width
    mouth_w = math.sqrt(
        (right_mouth[0] - left_mouth[0])**2 +
        (right_mouth[1] - left_mouth[1])**2
    )

    # Face width for normalization
    f_width = face_width(landmarks)
    if f_width < 0.01:
        return 0.0

    # Mouth to face ratio
    ratio = mouth_w / f_width

    # Scale: 0.35 = neutral, 0.50 = big smile
    smile = (ratio - 0.35) / 0.15

    return max(0.0, min(1.0, smile))


def interpolate_position(pos1: Tuple[float, float],
                         pos2: Tuple[float, float],
                         t: float) -> Tuple[float, float]:
    """
    Linear interpolation between two positions.

    Args:
        pos1: Starting position (x, y)
        pos2: Ending position (x, y)
        t: Interpolation factor (0-1)

    Returns:
        Interpolated position (x, y)
    """
    t = max(0.0, min(1.0, t))
    return (
        pos1[0] + (pos2[0] - pos1[0]) * t,
        pos1[1] + (pos2[1] - pos1[1]) * t
    )


def smooth_value(current: float, target: float, smoothing: float = 0.3) -> float:
    """
    Exponential smoothing for animation.

    Args:
        current: Current value
        target: Target value
        smoothing: Smoothing factor (0 = no smoothing, 1 = instant)

    Returns:
        Smoothed value
    """
    return current + (target - current) * smoothing
