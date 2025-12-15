"""
Tests for AR Filter Primitives Module

Tests the geometry generation functions in primitives.py.
NO OpenGL, NO hardware required.
All tests can run headless.

NOTE: All primitives now return TRIANGLE-BASED geometry for Core Profile compatibility.
"""

import pytest
import math
import numpy as np

# Import primitives functions
from ar_filter.primitives import (
    build_circle,
    build_filled_circle,
    build_horn,
    build_mask_outline,
    build_cat_ears,
    build_halo,
    build_neon_lines,
    build_zigzag_line,
    build_star,
    build_heart,
)


class TestBuildCircle:
    """Tests for build_circle function (triangle-based)."""

    def test_returns_numpy_array(self):
        """Should return numpy array."""
        result = build_circle((0.5, 0.5), 0.1)
        assert isinstance(result, np.ndarray)

    def test_correct_dtype(self):
        """Should return float32 array."""
        result = build_circle((0.5, 0.5), 0.1)
        assert result.dtype == np.float32

    def test_returns_triangles(self):
        """Should return triangle vertices (multiple of 6 per segment)."""
        segments = 16
        result = build_circle((0.5, 0.5), 0.1, segments=segments)
        # 6 vertices per segment (2 triangles)
        assert len(result) == segments * 6

    def test_vertices_form_ring(self):
        """Vertices should form a ring around center."""
        center = (0.5, 0.5)
        radius = 0.2
        result = build_circle(center, radius, segments=32)

        # Calculate distances from center
        distances = np.sqrt((result[:, 0] - center[0])**2 +
                           (result[:, 1] - center[1])**2)

        # Should have inner and outer edges near the radius
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        assert min_dist < radius  # Inner edge
        assert max_dist > radius  # Outer edge (with thickness)

    def test_circle_has_area(self):
        """Circle should have non-zero area."""
        result = build_circle((0.5, 0.5), 0.1, segments=8)

        # Check bounding box has area
        width = np.max(result[:, 0]) - np.min(result[:, 0])
        height = np.max(result[:, 1]) - np.min(result[:, 1])

        assert width > 0
        assert height > 0


class TestBuildFilledCircle:
    """Tests for build_filled_circle function."""

    def test_first_vertex_is_center(self):
        """First vertex should be the center."""
        center = (0.3, 0.7)
        result = build_filled_circle(center, 0.1)

        assert abs(result[0][0] - center[0]) < 0.001
        assert abs(result[0][1] - center[1]) < 0.001

    def test_correct_vertex_count(self):
        """Should have segments + 2 vertices (center + perimeter + closing)."""
        segments = 16
        result = build_filled_circle((0.5, 0.5), 0.1, segments=segments)
        assert len(result) == segments + 2


class TestBuildHorn:
    """Tests for build_horn function (triangle-based)."""

    def test_returns_numpy_array(self):
        """Should return numpy array."""
        result = build_horn((0.5, 0.5), 1.0, 0.0)
        assert isinstance(result, np.ndarray)

    def test_has_vertices(self):
        """Horn should have multiple vertices."""
        result = build_horn((0.5, 0.5), 1.0, 0.0)
        assert len(result) >= 6  # At least one triangle pair

    def test_scale_affects_size(self):
        """Larger scale should produce larger horn."""
        small = build_horn((0.5, 0.5), 0.5, 0.0)
        large = build_horn((0.5, 0.5), 2.0, 0.0)

        # Calculate bounding box heights
        small_height = np.max(small[:, 1]) - np.min(small[:, 1])
        large_height = np.max(large[:, 1]) - np.min(large[:, 1])

        assert large_height > small_height

    def test_returns_triangles(self):
        """Horn should return triangle vertices (multiple of 6)."""
        result = build_horn((0.5, 0.5), 1.0, 0.0)
        assert len(result) % 6 == 0


class TestBuildMaskOutline:
    """Tests for build_mask_outline function."""

    def test_empty_points_returns_empty(self):
        """Empty face points should return empty array."""
        result = build_mask_outline({})
        assert len(result) == 0

    def test_missing_keys_returns_empty(self):
        """Missing required keys should return empty array."""
        partial = {'nose_tip': (0.5, 0.5, 0.0)}
        result = build_mask_outline(partial)
        assert len(result) == 0

    def test_valid_points_returns_vertices(self):
        """Valid face points should return vertices."""
        face_points = {
            'left_eyebrow': (0.3, 0.4, 0.0),
            'right_eyebrow': (0.7, 0.4, 0.0),
            'left_eye_outer': (0.25, 0.45, 0.0),
            'right_eye_outer': (0.75, 0.45, 0.0),
            'nose_tip': (0.5, 0.55, 0.0),
            'forehead': (0.5, 0.35, 0.0),
        }

        result = build_mask_outline(face_points)
        assert len(result) > 0
        assert isinstance(result, np.ndarray)


class TestBuildCatEars:
    """Tests for build_cat_ears function."""

    def test_returns_two_arrays(self):
        """Should return tuple of two arrays (left, right ear)."""
        left, right = build_cat_ears((0.5, 0.3), 0.3)

        assert isinstance(left, np.ndarray)
        assert isinstance(right, np.ndarray)

    def test_ears_have_vertices(self):
        """Both ears should have vertices."""
        left, right = build_cat_ears((0.5, 0.3), 0.3)

        assert len(left) > 0
        assert len(right) > 0

    def test_ears_positioned_symmetrically(self):
        """Ears should be positioned on opposite sides of center."""
        forehead = (0.5, 0.3)
        left, right = build_cat_ears(forehead, 0.3)

        # Average X position of each ear
        left_avg_x = np.mean(left[:, 0])
        right_avg_x = np.mean(right[:, 0])

        assert left_avg_x < forehead[0]
        assert right_avg_x > forehead[0]


class TestBuildHalo:
    """Tests for build_halo function (triangle-based circles)."""

    def test_returns_two_circles(self):
        """Should return outer and inner circle arrays."""
        outer, inner = build_halo((0.5, 0.5), 0.2)

        assert isinstance(outer, np.ndarray)
        assert isinstance(inner, np.ndarray)

    def test_both_have_vertices(self):
        """Both circles should have triangle vertices."""
        outer, inner = build_halo((0.5, 0.5), 0.2)

        assert len(outer) > 0
        assert len(inner) > 0


class TestBuildNeonLines:
    """Tests for build_neon_lines function (triangle-based)."""

    def test_returns_list_of_arrays(self):
        """Should return list of line arrays."""
        result = build_neon_lines((0.0, 0.0), (1.0, 1.0), num_lines=3)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_each_line_is_triangles(self):
        """Each line should be 6 vertices (2 triangles for a quad)."""
        result = build_neon_lines((0.0, 0.0), (1.0, 1.0))

        for line in result:
            assert len(line) == 6  # Thick line = 2 triangles

    def test_zero_length_line_returns_empty(self):
        """Same start and end should return empty list."""
        result = build_neon_lines((0.5, 0.5), (0.5, 0.5))
        assert len(result) == 0


class TestBuildZigzagLine:
    """Tests for build_zigzag_line function (triangle-based)."""

    def test_returns_triangles(self):
        """Zigzag should return triangle vertices."""
        result = build_zigzag_line((0.0, 0.0), (1.0, 1.0))

        assert len(result) > 0
        assert len(result) % 6 == 0  # Multiple of 6 (triangles)

    def test_has_area(self):
        """Zigzag line should have non-zero area (thickness)."""
        result = build_zigzag_line((0.0, 0.0), (1.0, 0.0), amplitude=0.1)

        # Y should have variation from thickness
        y_min = np.min(result[:, 1])
        y_max = np.max(result[:, 1])

        assert y_max - y_min > 0

    def test_amplitude_affects_deviation(self):
        """Larger amplitude should create larger deviations."""
        small = build_zigzag_line((0.0, 0.0), (1.0, 0.0), amplitude=0.01)
        large = build_zigzag_line((0.0, 0.0), (1.0, 0.0), amplitude=0.1)

        # Calculate Y range for each
        small_range = np.max(small[:, 1]) - np.min(small[:, 1])
        large_range = np.max(large[:, 1]) - np.min(large[:, 1])

        assert large_range > small_range


class TestBuildStar:
    """Tests for build_star function (triangle-based)."""

    def test_returns_numpy_array(self):
        """Should return numpy array."""
        result = build_star((0.5, 0.5), 0.2)
        assert isinstance(result, np.ndarray)

    def test_returns_triangles(self):
        """Should return triangle vertices (multiple of 6)."""
        result = build_star((0.5, 0.5), 0.2, points=5)
        assert len(result) % 6 == 0

    def test_has_star_shape(self):
        """Star should have multiple points radiating from center."""
        center = (0.5, 0.5)
        result = build_star(center, 0.2, points=5)

        # Should cover an area around center
        width = np.max(result[:, 0]) - np.min(result[:, 0])
        height = np.max(result[:, 1]) - np.min(result[:, 1])

        assert width > 0.1
        assert height > 0.1


class TestBuildHeart:
    """Tests for build_heart function (triangle-based)."""

    def test_returns_numpy_array(self):
        """Should return numpy array."""
        result = build_heart((0.5, 0.5), 1.0)
        assert isinstance(result, np.ndarray)

    def test_returns_triangles(self):
        """Should return triangle vertices."""
        result = build_heart((0.5, 0.5), 1.0)
        assert len(result) % 6 == 0

    def test_centered_around_position(self):
        """Heart should be roughly centered around given position."""
        center = (0.3, 0.7)
        result = build_heart(center, 1.0)

        avg_x = np.mean(result[:, 0])
        avg_y = np.mean(result[:, 1])

        # Should be close to center (with some offset due to heart shape)
        assert abs(avg_x - center[0]) < 0.15
        assert abs(avg_y - center[1]) < 0.25


class TestVertexValidity:
    """Tests for vertex array validity."""

    def test_all_primitives_have_2d_vertices(self):
        """All primitives should produce 2D vertices."""
        primitives = [
            build_circle((0.5, 0.5), 0.1),
            build_filled_circle((0.5, 0.5), 0.1),
            build_horn((0.5, 0.5), 1.0, 0.0),
            build_zigzag_line((0.0, 0.0), (1.0, 1.0)),
            build_star((0.5, 0.5), 0.2),
            build_heart((0.5, 0.5), 1.0),
        ]

        for prim in primitives:
            assert prim.ndim == 2
            assert prim.shape[1] == 2

    def test_no_nan_values(self):
        """Primitives should not contain NaN values."""
        primitives = [
            build_circle((0.5, 0.5), 0.1),
            build_horn((0.5, 0.5), 1.0, 0.5),
            build_star((0.5, 0.5), 0.2),
        ]

        for prim in primitives:
            assert not np.any(np.isnan(prim))

    def test_no_infinite_values(self):
        """Primitives should not contain infinite values."""
        primitives = [
            build_circle((0.5, 0.5), 0.1),
            build_horn((0.5, 0.5), 1.0, 0.5),
            build_star((0.5, 0.5), 0.2),
        ]

        for prim in primitives:
            assert not np.any(np.isinf(prim))


class TestScaling:
    """Tests for proper scaling of primitives."""

    def test_circle_scales_with_radius(self):
        """Larger radius should produce larger circle."""
        center = (0.5, 0.5)
        small = build_circle(center, 0.1)
        large = build_circle(center, 0.2)

        small_size = np.max(small[:, 0]) - np.min(small[:, 0])
        large_size = np.max(large[:, 0]) - np.min(large[:, 0])

        assert large_size > small_size

    def test_star_scales_with_radius(self):
        """Star size should scale with outer radius."""
        center = (0.5, 0.5)
        small = build_star(center, 0.1)
        large = build_star(center, 0.3)

        small_size = np.max(small[:, 0]) - np.min(small[:, 0])
        large_size = np.max(large[:, 0]) - np.min(large[:, 0])

        assert large_size > small_size
