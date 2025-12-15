import math
import sys
import types

import pytest

_original_cv2 = sys.modules.get("cv2")
sys.modules["cv2"] = types.SimpleNamespace(
    FILLED=0,
    FONT_HERSHEY_SIMPLEX=0,
    LINE_AA=0,
    INTER_LINEAR=0,
    cvtColor=lambda *args, **kwargs: None,
    COLOR_BGR2RGB=0,
    circle=lambda *args, **kwargs: None,
    line=lambda *args, **kwargs: None,
    putText=lambda *args, **kwargs: None,
)

from mediapipe.framework.formats import landmark_pb2
from ar_filter.metrics import (
    face_width,
    halo_radius,
    head_position,
    mouth_open_ratio,
    mouth_reference_points,
    smooth_value,
)

if _original_cv2 is None:
    sys.modules.pop("cv2", None)
else:
    sys.modules["cv2"] = _original_cv2


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


def build_mediapipe_landmark_list():
    mp_list = landmark_pb2.NormalizedLandmarkList()
    mp_list.landmark.extend(
        [landmark_pb2.NormalizedLandmark(x=0.0, y=0.0) for _ in range(500)]
    )
    mp_list.landmark[234].x, mp_list.landmark[234].y = 0.2, 0.5
    mp_list.landmark[454].x, mp_list.landmark[454].y = 0.8, 0.5
    mp_list.landmark[13].x, mp_list.landmark[13].y = 0.5, 0.45
    mp_list.landmark[14].x, mp_list.landmark[14].y = 0.5, 0.55
    mp_list.landmark[10].x, mp_list.landmark[10].y = 0.52, 0.3
    mp_list.landmark[61].x, mp_list.landmark[61].y = 0.45, 0.52
    mp_list.landmark[291].x, mp_list.landmark[291].y = 0.55, 0.52
    return mp_list


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


def test_metrics_accepts_plain_list():
    landmarks = build_landmarks()
    assert face_width(landmarks) > 0
    assert mouth_open_ratio(landmarks) > 0


def test_metrics_accepts_mediapipe_landmark_list_mock():
    class LandmarkList:
        def __init__(self, points):
            self.landmark = points

    mp_landmarks = LandmarkList(build_landmarks())
    assert face_width(mp_landmarks) > 0
    assert mouth_open_ratio(mp_landmarks) > 0


def test_metrics_accepts_real_mediapipe_landmark_list():
    mp_landmarks = build_mediapipe_landmark_list()
    assert face_width(mp_landmarks) > 0
    assert mouth_open_ratio(mp_landmarks) > 0


def test_metrics_rejects_unsupported_landmark_container():
    class Unsupported:
        pass

    with pytest.raises(TypeError):
        face_width(Unsupported())
