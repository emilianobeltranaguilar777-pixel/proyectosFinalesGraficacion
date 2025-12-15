import math

import pytest

from ar_filter.metrics import face_width, halo_radius, mouth_open_ratio, smooth_value


class Landmark:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def build_landmarks():
    points = [Landmark(0.0, 0.0) for _ in range(500)]
    points[234] = Landmark(0.2, 0.5)
    points[454] = Landmark(0.8, 0.5)
    points[13] = Landmark(0.5, 0.45)
    points[14] = Landmark(0.5, 0.55)
    return points


def test_face_width_uses_cheek_distance():
    landmarks = build_landmarks()
    assert math.isclose(face_width(landmarks), 0.6)


def test_halo_radius_scales_face_width():
    landmarks = build_landmarks()
    assert math.isclose(halo_radius(landmarks, scale=1.0), face_width(landmarks))


def test_mouth_open_ratio_normalized_by_face():
    landmarks = build_landmarks()
    ratio = mouth_open_ratio(landmarks)
    assert pytest.approx(0.166, rel=1e-2) == ratio


def test_smooth_value_moves_toward_target():
    assert smooth_value(0.0, 1.0, 0.5) == 0.5
