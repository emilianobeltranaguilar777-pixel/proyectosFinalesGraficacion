"""Pure utility metrics for AR overlay logic."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple
import math


LANDMARK_LEFT_CHEEK = 234
LANDMARK_RIGHT_CHEEK = 454
LANDMARK_UPPER_LIP = 13
LANDMARK_LOWER_LIP = 14
LANDMARK_FOREHEAD = 10
LANDMARK_MOUTH_LEFT = 61
LANDMARK_MOUTH_RIGHT = 291


def _point(landmarks: Sequence, index: int) -> Tuple[float, float]:
    landmark = landmarks[index]
    return float(landmark.x), float(landmark.y)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def face_width(landmarks: Sequence) -> float:
    """Returns the normalized face width using cheek landmarks."""
    left = _point(landmarks, LANDMARK_LEFT_CHEEK)
    right = _point(landmarks, LANDMARK_RIGHT_CHEEK)
    return _distance(left, right)


def mouth_open_ratio(landmarks: Sequence) -> float:
    """Computes mouth opening normalized by face width."""
    top = _point(landmarks, LANDMARK_UPPER_LIP)
    bottom = _point(landmarks, LANDMARK_LOWER_LIP)
    opening = _distance(top, bottom)
    width = max(face_width(landmarks), 1e-6)
    return opening / width


def halo_radius(landmarks: Sequence, scale: float = 1.1) -> float:
    """Determines halo radius from face width."""
    return face_width(landmarks) * scale


def normalized_to_viewport(point: Tuple[float, float]) -> Tuple[float, float]:
    """Converts normalized landmark coordinates to clip-space viewport."""
    x, y = point
    return (x * 2.0 - 1.0, 1.0 - y * 2.0)


def head_position(landmarks: Sequence) -> Tuple[float, float]:
    """Returns the viewport position of the forehead landmark."""
    return normalized_to_viewport(_point(landmarks, LANDMARK_FOREHEAD))


def mouth_corners(landmarks: Sequence) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Returns viewport positions for the left and right mouth corners."""
    left = normalized_to_viewport(_point(landmarks, LANDMARK_MOUTH_LEFT))
    right = normalized_to_viewport(_point(landmarks, LANDMARK_MOUTH_RIGHT))
    return left, right


def smooth_value(current: float, target: float, factor: float) -> float:
    """Simple linear interpolation for smooth transitions."""
    return current + (target - current) * factor
