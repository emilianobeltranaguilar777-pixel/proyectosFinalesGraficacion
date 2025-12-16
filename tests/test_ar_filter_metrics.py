"""
Tests for AR filter metrics module.

These tests verify pure math functions without any OpenGL or camera dependencies.
"""

import pytest
import math
from ar_filter.metrics import (
    clamp,
    face_width,
    halo_radius,
    mouth_openness,
    face_center,
    lerp,
    smooth_value,
    halo_sphere_positions,
    mouth_rect_scale,
    mouth_rect_color,
)


class TestClamp:
    """Tests for clamp function."""

    def test_clamp_within_range(self):
        """Value within range should be unchanged."""
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_clamp_below_min(self):
        """Value below min should return min."""
        assert clamp(-5.0, 0.0, 10.0) == 0.0

    def test_clamp_above_max(self):
        """Value above max should return max."""
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_clamp_at_boundaries(self):
        """Values at boundaries should be unchanged."""
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0

    def test_clamp_negative_range(self):
        """Clamp should work with negative ranges."""
        assert clamp(-5.0, -10.0, -1.0) == -5.0
        assert clamp(0.0, -10.0, -1.0) == -1.0


class TestFaceWidth:
    """Tests for face_width function."""

    def test_face_width_empty_landmarks(self):
        """Empty landmarks should return 0."""
        assert face_width([]) == 0.0

    def test_face_width_insufficient_landmarks(self):
        """Insufficient landmarks should return 0."""
        landmarks = [(0.1, 0.1, 0.0)] * 100  # Not enough for index 454
        assert face_width(landmarks) == 0.0

    def test_face_width_horizontal(self):
        """Calculate width for horizontal face."""
        # Create minimal landmarks with left and right cheek positions
        landmarks = [(0.0, 0.0, 0.0)] * 500
        landmarks[234] = (0.2, 0.5, 0.0)  # Left cheek
        landmarks[454] = (0.8, 0.5, 0.0)  # Right cheek

        width = face_width(landmarks)
        assert abs(width - 0.6) < 0.001

    def test_face_width_diagonal(self):
        """Calculate width for diagonal face positions."""
        landmarks = [(0.0, 0.0, 0.0)] * 500
        landmarks[234] = (0.0, 0.0, 0.0)
        landmarks[454] = (0.3, 0.4, 0.0)

        width = face_width(landmarks)
        expected = math.sqrt(0.3**2 + 0.4**2)  # 0.5
        assert abs(width - expected) < 0.001

    def test_face_width_custom_indices(self):
        """Custom indices should be used correctly."""
        landmarks = [(0.0, 0.0, 0.0)] * 100
        landmarks[10] = (0.1, 0.5, 0.0)
        landmarks[20] = (0.9, 0.5, 0.0)

        width = face_width(landmarks, left_idx=10, right_idx=20)
        assert abs(width - 0.8) < 0.001


class TestHaloRadius:
    """Tests for halo_radius function."""

    def test_halo_radius_basic(self):
        """Basic halo radius calculation."""
        radius = halo_radius(0.2)
        assert abs(radius - 0.3) < 0.001

    def test_halo_radius_clamped_min(self):
        """Very small face should clamp to minimum."""
        radius = halo_radius(0.01)
        assert radius == 0.05

    def test_halo_radius_clamped_max(self):
        """Very large face should clamp to maximum."""
        radius = halo_radius(1.0)
        assert radius == 0.5

    def test_halo_radius_custom_scale(self):
        """Custom scale factor should be applied."""
        radius = halo_radius(0.2, scale_factor=2.0)
        assert abs(radius - 0.4) < 0.001


class TestMouthOpenness:
    """Tests for mouth_openness function."""

    def test_mouth_openness_empty(self):
        """Empty landmarks should return 0."""
        assert mouth_openness([]) == 0.0

    def test_mouth_openness_closed(self):
        """Closed mouth should return ~0."""
        landmarks = [(0.0, 0.0, 0.0)] * 500
        landmarks[13] = (0.5, 0.500, 0.0)  # Top lip
        landmarks[14] = (0.5, 0.510, 0.0)  # Bottom lip (very close)

        openness = mouth_openness(landmarks)
        assert openness < 0.01  # Allow small floating point tolerance

    def test_mouth_openness_wide_open(self):
        """Wide open mouth should return ~1."""
        landmarks = [(0.0, 0.0, 0.0)] * 500
        landmarks[13] = (0.5, 0.50, 0.0)  # Top lip
        landmarks[14] = (0.5, 0.57, 0.0)  # Bottom lip (far apart)

        openness = mouth_openness(landmarks)
        assert openness == 1.0

    def test_mouth_openness_partial(self):
        """Partially open mouth should return intermediate value."""
        landmarks = [(0.0, 0.0, 0.0)] * 500
        landmarks[13] = (0.5, 0.50, 0.0)  # Top lip
        landmarks[14] = (0.5, 0.535, 0.0)  # Bottom lip (half open)

        openness = mouth_openness(landmarks)
        assert 0.3 < openness < 0.7


class TestFaceCenter:
    """Tests for face_center function."""

    def test_face_center_empty(self):
        """Empty landmarks should return default center."""
        center = face_center([])
        assert center == (0.5, 0.5, 0.0)

    def test_face_center_valid(self):
        """Valid landmarks should return nose position."""
        landmarks = [(0.0, 0.0, 0.0)] * 500
        landmarks[1] = (0.45, 0.55, 0.1)

        center = face_center(landmarks)
        assert center == (0.45, 0.55, 0.1)


class TestLerp:
    """Tests for lerp function."""

    def test_lerp_start(self):
        """t=0 should return start value."""
        assert lerp(0.0, 10.0, 0.0) == 0.0

    def test_lerp_end(self):
        """t=1 should return end value."""
        assert lerp(0.0, 10.0, 1.0) == 10.0

    def test_lerp_middle(self):
        """t=0.5 should return middle value."""
        assert lerp(0.0, 10.0, 0.5) == 5.0

    def test_lerp_quarter(self):
        """t=0.25 should return quarter value."""
        assert lerp(0.0, 10.0, 0.25) == 2.5

    def test_lerp_clamped(self):
        """t outside [0,1] should be clamped."""
        assert lerp(0.0, 10.0, -1.0) == 0.0
        assert lerp(0.0, 10.0, 2.0) == 10.0


class TestSmoothValue:
    """Tests for smooth_value function."""

    def test_smooth_value_no_smoothing(self):
        """Full smoothing should snap to target."""
        result = smooth_value(0.0, 10.0, smoothing=1.0)
        assert result == 10.0

    def test_smooth_value_full_smoothing(self):
        """No smoothing should stay at current."""
        result = smooth_value(5.0, 10.0, smoothing=0.0)
        assert result == 5.0

    def test_smooth_value_partial(self):
        """Partial smoothing should move towards target."""
        result = smooth_value(0.0, 10.0, smoothing=0.3)
        assert abs(result - 3.0) < 0.001


class TestHaloSpherePositions:
    """Tests for halo_sphere_positions function."""

    def test_halo_positions_empty(self):
        """Zero spheres should return empty list."""
        positions = halo_sphere_positions((0.5, 0.5, 0.0), 0.2, 0, 0.0)
        assert positions == []

    def test_halo_positions_count(self):
        """Should return correct number of positions."""
        positions = halo_sphere_positions((0.5, 0.5, 0.0), 0.2, 8, 0.0)
        assert len(positions) == 8

    def test_halo_positions_radius(self):
        """Positions should be at approximately correct radius."""
        center = (0.5, 0.5, 0.0)
        radius = 0.2
        positions = halo_sphere_positions(center, radius, 4, 0.0)

        for pos in positions:
            dx = pos[0] - center[0]
            # Y has offset, Z is scaled, check X spread
            assert abs(dx) <= radius

    def test_halo_positions_rotation(self):
        """Different rotation angles should produce different positions."""
        pos1 = halo_sphere_positions((0.5, 0.5, 0.0), 0.2, 4, 0.0)
        pos2 = halo_sphere_positions((0.5, 0.5, 0.0), 0.2, 4, math.pi / 4)

        # First position X should be different
        assert pos1[0][0] != pos2[0][0]


class TestMouthRectScale:
    """Tests for mouth_rect_scale function."""

    def test_rect_scale_closed(self):
        """Closed mouth should return min scale."""
        scale = mouth_rect_scale(0.0)
        assert scale == 0.5

    def test_rect_scale_open(self):
        """Open mouth should return max scale."""
        scale = mouth_rect_scale(1.0)
        assert scale == 2.0

    def test_rect_scale_partial(self):
        """Partial openness should interpolate."""
        scale = mouth_rect_scale(0.5)
        assert abs(scale - 1.25) < 0.001

    def test_rect_scale_custom_range(self):
        """Custom scale range should be used."""
        scale = mouth_rect_scale(0.5, min_scale=1.0, max_scale=3.0)
        assert scale == 2.0


class TestMouthRectColor:
    """Tests for mouth_rect_color function."""

    def test_rect_color_closed(self):
        """Closed mouth should return closed color."""
        color = mouth_rect_color(0.0)
        assert color == (0.2, 0.6, 1.0)

    def test_rect_color_open(self):
        """Open mouth should return open color."""
        color = mouth_rect_color(1.0)
        assert color == (1.0, 0.3, 0.5)

    def test_rect_color_partial(self):
        """Partial openness should interpolate colors."""
        color = mouth_rect_color(0.5)
        assert abs(color[0] - 0.6) < 0.001
        assert abs(color[1] - 0.45) < 0.001
        assert abs(color[2] - 0.75) < 0.001

    def test_rect_color_custom(self):
        """Custom colors should be used."""
        color = mouth_rect_color(1.0,
                                 closed_color=(0.0, 0.0, 0.0),
                                 open_color=(1.0, 1.0, 1.0))
        assert color == (1.0, 1.0, 1.0)
