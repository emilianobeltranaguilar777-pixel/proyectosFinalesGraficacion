import math

import pytest

from ar_filter.metrics import (
    face_width,
    halo_radius,
    head_position,
    mouth_corners,
    mouth_open_ratio,
    normalized_to_viewport,
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
    points[10] = Landmark(0.5, 0.25)
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


def test_normalized_to_viewport_maps_range():
    assert normalized_to_viewport((0.0, 0.0)) == (-1.0, 1.0)
    assert normalized_to_viewport((1.0, 1.0)) == (1.0, -1.0)


def test_head_position_uses_forehead_landmark():
    landmarks = build_landmarks()
    assert head_position(landmarks) == (0.0, 0.5)


def test_mouth_corners_from_landmarks():
    landmarks = build_landmarks()
    left, right = mouth_corners(landmarks)
    assert left == (-0.09999999999999998, -0.040000000000000036)
    assert right == (0.10000000000000009, -0.040000000000000036)


def test_smooth_value_moves_toward_target():
    assert smooth_value(0.0, 1.0, 0.5) == 0.5
