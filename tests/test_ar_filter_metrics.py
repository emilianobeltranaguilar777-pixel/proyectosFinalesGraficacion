import math

import pytest

from ar_filter.metrics import (
    face_width,
    halo_radius,
    head_position,
    mouth_open_ratio,
    mouth_reference_points,
    smooth_value,
)


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
    points[10] = Landmark(0.52, 0.3)
    points[61] = Landmark(0.45, 0.52)
    points[291] = Landmark(0.55, 0.52)
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


def test_head_position_uses_forehead_landmark():
    landmarks = build_landmarks()
    x, y = head_position(landmarks)
    assert x == pytest.approx(0.52)
    assert y == pytest.approx(0.3)


def test_mouth_reference_points_return_corners_and_center():
    landmarks = build_landmarks()
    left, right, center_y = mouth_reference_points(landmarks)
    assert left == (pytest.approx(0.45), pytest.approx(0.52))
    assert right == (pytest.approx(0.55), pytest.approx(0.52))
    assert center_y == pytest.approx(0.5)
