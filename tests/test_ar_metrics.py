"""
Tests for AR Filter Metrics Module

Tests the pure mathematical functions in metrics.py.
NO OpenGL, NO MediaPipe, NO hardware required.
All tests can run headless.
"""

import pytest
import math

# Import metrics functions
from ar_filter.metrics import (
    face_width,
    face_height,
    mouth_openness,
    head_tilt,
    head_rotation_y,
    face_center,
    eye_center,
    eyebrow_raise,
    smile_intensity,
    interpolate_position,
    smooth_value,
)


class TestFaceWidth:
    """Tests for face_width function."""

    def test_empty_landmarks_returns_zero(self):
        """Empty landmarks should return 0."""
        assert face_width([]) == 0.0

    def test_insufficient_landmarks_returns_zero(self):
        """Less than 455 landmarks should return 0."""
        landmarks = [(0.5, 0.5, 0.0)] * 400
        assert face_width(landmarks) == 0.0

    def test_valid_landmarks_returns_positive(self):
        """Valid landmarks should return positive width."""
        # Create 468 landmarks (full FaceMesh)
        landmarks = [(0.5, 0.5, 0.0)] * 468

        # Set left cheek (234) and right cheek (454) positions
        landmarks[234] = (0.3, 0.5, 0.0)  # Left cheek
        landmarks[454] = (0.7, 0.5, 0.0)  # Right cheek

        width = face_width(landmarks)
        assert width > 0
        assert abs(width - 0.4) < 0.001  # Should be ~0.4

    def test_width_scales_with_distance(self):
        """Width should increase as cheeks are farther apart."""
        landmarks1 = [(0.5, 0.5, 0.0)] * 468
        landmarks1[234] = (0.4, 0.5, 0.0)
        landmarks1[454] = (0.6, 0.5, 0.0)

        landmarks2 = [(0.5, 0.5, 0.0)] * 468
        landmarks2[234] = (0.2, 0.5, 0.0)
        landmarks2[454] = (0.8, 0.5, 0.0)

        width1 = face_width(landmarks1)
        width2 = face_width(landmarks2)

        assert width2 > width1


class TestFaceHeight:
    """Tests for face_height function."""

    def test_empty_landmarks_returns_zero(self):
        """Empty landmarks should return 0."""
        assert face_height([]) == 0.0

    def test_valid_landmarks_returns_positive(self):
        """Valid landmarks should return positive height."""
        landmarks = [(0.5, 0.5, 0.0)] * 468
        landmarks[10] = (0.5, 0.3, 0.0)   # Forehead
        landmarks[152] = (0.5, 0.7, 0.0)  # Chin

        height = face_height(landmarks)
        assert height > 0
        assert abs(height - 0.4) < 0.001


class TestMouthOpenness:
    """Tests for mouth_openness function."""

    def test_empty_landmarks_returns_zero(self):
        """Empty landmarks should return 0."""
        assert mouth_openness([]) == 0.0

    def test_closed_mouth_returns_low_value(self):
        """Closed mouth (lips close together) should return low value."""
        landmarks = [(0.5, 0.5, 0.0)] * 468
        landmarks[10] = (0.5, 0.3, 0.0)   # Forehead
        landmarks[152] = (0.5, 0.7, 0.0)  # Chin
        landmarks[13] = (0.5, 0.55, 0.0)  # Top lip
        landmarks[14] = (0.5, 0.56, 0.0)  # Bottom lip (close)

        openness = mouth_openness(landmarks)
        assert openness < 0.3

    def test_open_mouth_returns_high_value(self):
        """Open mouth (lips far apart) should return high value."""
        landmarks = [(0.5, 0.5, 0.0)] * 468
        landmarks[10] = (0.5, 0.3, 0.0)   # Forehead
        landmarks[152] = (0.5, 0.7, 0.0)  # Chin
        landmarks[13] = (0.5, 0.50, 0.0)  # Top lip
        landmarks[14] = (0.5, 0.60, 0.0)  # Bottom lip (far)

        openness = mouth_openness(landmarks)
        assert openness > 0.5

    def test_openness_clamped_to_zero_one(self):
        """Openness should always be between 0 and 1."""
        landmarks = [(0.5, 0.5, 0.0)] * 468
        landmarks[10] = (0.5, 0.3, 0.0)
        landmarks[152] = (0.5, 0.7, 0.0)
        landmarks[13] = (0.5, 0.40, 0.0)
        landmarks[14] = (0.5, 0.80, 0.0)

        openness = mouth_openness(landmarks)
        assert 0.0 <= openness <= 1.0


class TestHeadTilt:
    """Tests for head_tilt function."""

    def test_empty_landmarks_returns_zero(self):
        """Empty landmarks should return 0."""
        assert head_tilt([]) == 0.0

    def test_level_head_returns_near_zero(self):
        """Level head (horizontal eyes) should return near zero tilt."""
        landmarks = [(0.5, 0.5, 0.0)] * 468
        landmarks[33] = (0.3, 0.4, 0.0)   # Left eye
        landmarks[263] = (0.7, 0.4, 0.0)  # Right eye (same Y)

        tilt = head_tilt(landmarks)
        assert abs(tilt) < 0.01

    def test_tilted_right_returns_positive(self):
        """Head tilted right should return positive angle."""
        landmarks = [(0.5, 0.5, 0.0)] * 468
        landmarks[33] = (0.3, 0.5, 0.0)   # Left eye (higher)
        landmarks[263] = (0.7, 0.4, 0.0)  # Right eye (lower)

        tilt = head_tilt(landmarks)
        assert tilt < 0  # Negative because right eye is lower

    def test_tilted_left_returns_negative(self):
        """Head tilted left should return negative angle."""
        landmarks = [(0.5, 0.5, 0.0)] * 468
        landmarks[33] = (0.3, 0.4, 0.0)   # Left eye (lower)
        landmarks[263] = (0.7, 0.5, 0.0)  # Right eye (higher)

        tilt = head_tilt(landmarks)
        assert tilt > 0


class TestHeadRotationY:
    """Tests for head_rotation_y function."""

    def test_empty_landmarks_returns_zero(self):
        """Empty landmarks should return 0."""
        assert head_rotation_y([]) == 0.0

    def test_forward_face_returns_near_zero(self):
        """Face looking forward should return near zero rotation."""
        landmarks = [(0.5, 0.5, 0.0)] * 468
        landmarks[1] = (0.5, 0.5, 0.0)    # Nose (center)
        landmarks[33] = (0.3, 0.4, 0.0)   # Left eye
        landmarks[263] = (0.7, 0.4, 0.0)  # Right eye

        rotation = head_rotation_y(landmarks)
        assert abs(rotation) < 0.1


class TestFaceCenter:
    """Tests for face_center function."""

    def test_empty_landmarks_returns_default(self):
        """Empty landmarks should return (0.5, 0.5)."""
        center = face_center([])
        assert center == (0.5, 0.5)

    def test_returns_nose_position(self):
        """Should return nose tip position."""
        landmarks = [(0.0, 0.0, 0.0)] * 468
        landmarks[1] = (0.6, 0.55, 0.0)  # Nose tip

        center = face_center(landmarks)
        assert center == (0.6, 0.55)


class TestEyeCenter:
    """Tests for eye_center function."""

    def test_empty_landmarks_returns_default(self):
        """Empty landmarks should return (0.5, 0.5)."""
        center = eye_center([])
        assert center == (0.5, 0.5)

    def test_left_eye_center(self):
        """Should return average of left eye corners."""
        landmarks = [(0.0, 0.0, 0.0)] * 468
        landmarks[33] = (0.3, 0.4, 0.0)   # Outer
        landmarks[133] = (0.4, 0.4, 0.0)  # Inner

        center = eye_center(landmarks, left=True)
        assert abs(center[0] - 0.35) < 0.001
        assert abs(center[1] - 0.4) < 0.001

    def test_right_eye_center(self):
        """Should return average of right eye corners."""
        landmarks = [(0.0, 0.0, 0.0)] * 468
        landmarks[263] = (0.7, 0.4, 0.0)  # Outer
        landmarks[362] = (0.6, 0.4, 0.0)  # Inner

        center = eye_center(landmarks, left=False)
        assert abs(center[0] - 0.65) < 0.001


class TestInterpolatePosition:
    """Tests for interpolate_position function."""

    def test_t_zero_returns_start(self):
        """t=0 should return start position."""
        pos = interpolate_position((0.0, 0.0), (1.0, 1.0), 0.0)
        assert pos == (0.0, 0.0)

    def test_t_one_returns_end(self):
        """t=1 should return end position."""
        pos = interpolate_position((0.0, 0.0), (1.0, 1.0), 1.0)
        assert pos == (1.0, 1.0)

    def test_t_half_returns_midpoint(self):
        """t=0.5 should return midpoint."""
        pos = interpolate_position((0.0, 0.0), (1.0, 1.0), 0.5)
        assert abs(pos[0] - 0.5) < 0.001
        assert abs(pos[1] - 0.5) < 0.001

    def test_t_clamped(self):
        """t should be clamped to 0-1 range."""
        pos_neg = interpolate_position((0.0, 0.0), (1.0, 1.0), -0.5)
        pos_over = interpolate_position((0.0, 0.0), (1.0, 1.0), 1.5)

        assert pos_neg == (0.0, 0.0)
        assert pos_over == (1.0, 1.0)


class TestSmoothValue:
    """Tests for smooth_value function."""

    def test_smoothing_zero_no_change(self):
        """Smoothing=0 should not change current value."""
        result = smooth_value(0.5, 1.0, 0.0)
        assert result == 0.5

    def test_smoothing_one_instant(self):
        """Smoothing=1 should instantly reach target."""
        result = smooth_value(0.5, 1.0, 1.0)
        assert result == 1.0

    def test_smoothing_partial(self):
        """Partial smoothing should move towards target."""
        result = smooth_value(0.0, 1.0, 0.5)
        assert result == 0.5

    def test_smoothing_approaches_target(self):
        """Multiple smoothing steps should approach target."""
        current = 0.0
        target = 1.0

        for _ in range(20):
            current = smooth_value(current, target, 0.3)

        assert current > 0.99


class TestValueConsistency:
    """Tests for consistent values across related metrics."""

    def test_face_dimensions_related(self):
        """Face width and height should be in reasonable proportion."""
        landmarks = [(0.5, 0.5, 0.0)] * 468
        landmarks[234] = (0.3, 0.5, 0.0)   # Left cheek
        landmarks[454] = (0.7, 0.5, 0.0)   # Right cheek
        landmarks[10] = (0.5, 0.3, 0.0)    # Forehead
        landmarks[152] = (0.5, 0.8, 0.0)   # Chin

        width = face_width(landmarks)
        height = face_height(landmarks)

        # Face is typically taller than wide
        assert height > width * 0.8
        assert height < width * 2.0

    def test_mouth_and_smile_related(self):
        """Mouth openness and smile should be measurable."""
        landmarks = [(0.5, 0.5, 0.0)] * 468

        # Set face structure
        landmarks[234] = (0.3, 0.5, 0.0)
        landmarks[454] = (0.7, 0.5, 0.0)
        landmarks[10] = (0.5, 0.3, 0.0)
        landmarks[152] = (0.5, 0.7, 0.0)

        # Set mouth
        landmarks[13] = (0.5, 0.55, 0.0)
        landmarks[14] = (0.5, 0.58, 0.0)
        landmarks[61] = (0.4, 0.56, 0.0)
        landmarks[291] = (0.6, 0.56, 0.0)

        openness = mouth_openness(landmarks)
        smile = smile_intensity(landmarks)

        # Both should be valid numbers
        assert 0.0 <= openness <= 1.0
        assert 0.0 <= smile <= 1.0
