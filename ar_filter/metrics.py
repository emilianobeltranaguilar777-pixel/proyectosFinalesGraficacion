"""Pure utility metrics for AR overlay logic."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple
import math

try:  # MediaPipe may be unavailable in headless test environments
    from mediapipe.framework.formats import landmark_pb2
except Exception:  # pragma: no cover - fallback for environments without MediaPipe
    landmark_pb2 = None  # type: ignore


LANDMARK_LEFT_CHEEK = 234
LANDMARK_RIGHT_CHEEK = 454
LANDMARK_UPPER_LIP = 13
LANDMARK_LOWER_LIP = 14
LANDMARK_FOREHEAD = 10
LANDMARK_MOUTH_LEFT = 61
LANDMARK_MOUTH_RIGHT = 291


def _as_landmark_list(landmarks: Sequence) -> Sequence:
    """Normalizes MediaPipe landmark containers into a sequence."""
    if landmark_pb2 is not None and isinstance(landmarks, landmark_pb2.NormalizedLandmarkList):
        return list(landmarks.landmark)
    if hasattr(landmarks, "landmark"):
        return list(landmarks.landmark)
    if isinstance(landmarks, (list, tuple)):
        return landmarks
    raise TypeError("Unsupported landmark container type")


def _point(landmarks: Sequence, index: int) -> Tuple[float, float]:
    normalized = _as_landmark_list(landmarks)
    landmark = normalized[index]
    return float(landmark.x), float(landmark.y)


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def face_width(landmarks: Sequence) -> float:
    """Returns the normalized face width using cheek landmarks."""
    normalized = _as_landmark_list(landmarks)
    left = _point(normalized, LANDMARK_LEFT_CHEEK)
    right = _point(normalized, LANDMARK_RIGHT_CHEEK)
    return _distance(left, right)


def mouth_open_ratio(landmarks: Sequence) -> float:
    """Computes mouth opening normalized by face width."""
    normalized = _as_landmark_list(landmarks)
    top = _point(normalized, LANDMARK_UPPER_LIP)
    bottom = _point(normalized, LANDMARK_LOWER_LIP)
    opening = _distance(top, bottom)
    width = max(face_width(normalized), 1e-6)
    return opening / width


def halo_radius(landmarks: Sequence, scale: float = 1.1) -> float:
    """Determines halo radius from face width."""
    normalized = _as_landmark_list(landmarks)
    return face_width(normalized) * scale


def head_position(landmarks: Sequence) -> Tuple[float, float]:
    """Returns a normalized anchor position near the forehead."""
    normalized = _as_landmark_list(landmarks)
    return _point(normalized, LANDMARK_FOREHEAD)


def mouth_reference_points(landmarks: Sequence) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Provides mouth corner positions and vertical center for anchoring quads."""
    normalized = _as_landmark_list(landmarks)
    left = _point(normalized, LANDMARK_MOUTH_LEFT)
    right = _point(normalized, LANDMARK_MOUTH_RIGHT)
    top = _point(normalized, LANDMARK_UPPER_LIP)
    bottom = _point(normalized, LANDMARK_LOWER_LIP)
    center_y = (top[1] + bottom[1]) * 0.5
    return left, right, center_y


def smooth_value(current: float, target: float, factor: float) -> float:
    """Simple linear interpolation for smooth transitions."""
    return current + (target - current) * factor
