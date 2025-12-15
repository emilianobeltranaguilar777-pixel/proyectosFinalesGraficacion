"""
Tests for AR Filter Primitives Module

Tests the geometry generation functions in primitives.py.
NO OpenGL, NO hardware required.
All tests can run headless.
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
    """Tests for build_circle function."""

    def test_returns_numpy_array(self):
        """Should return numpy array."""
        result = build_circle((0.5, 0.5), 0.1)
        assert isinstance(result, np.ndarray)

    def test_correct_dtype(self):
        """Should return float32 array."""
        result = build_circle((0.5, 0.5), 0.1)
        assert result.dtype == np.float32

    def test_correct_segment_count(self):
        """Should have correct number of vertices."""
        segments = 16
        result = build_circle((0.5, 0.5), 0.1, segments=segments)
        assert len(result) == segments

    def test_vertices_at_correct_radius(self):
        """All vertices should be at specified radius from center."""
        center = (0.5, 0.5)
        radius = 0.2
        result = build_circle(center, radius, segments=32)

        for vertex in result:
            dist = math.sqrt((vertex[0] - center[0])**2 + (vertex[1] - center[1])**2)
            assert abs(dist - radius) < 0.001

    def test_positive_radius(self):
        """Vertices should form valid circle with positive radius."""
        result = build_circle((0.0, 0.0), 1.0, segments=4)

        # Should have 4 points
        assert len(result) == 4

        # Check corners of unit circle
        expected_points = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for i, (ex, ey) in enumerate(expected_points):
            assert abs(result[i][0] - ex) < 0.001
            assert abs(result[i][1] - ey) < 0.001


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
    """Tests for build_horn function."""

    def test_returns_numpy_array(self):
        """Should return numpy array."""
        result = build_horn((0.5, 0.5), 1.0, 0.0)
        assert isinstance(result, np.ndarray)

    def test_starts_at_base_position(self):
        """First vertex should be at base position."""
        base = (0.3, 0.4)
        result = build_horn(base, 1.0, 0.0)

        assert abs(result[0][0] - base[0]) < 0.001
        assert abs(result[0][1] - base[1]) < 0.001

    def test_scale_affects_size(self):
        """Larger scale should produce larger horn."""
        small = build_horn((0.5, 0.5), 0.5, 0.0)
        large = build_horn((0.5, 0.5), 2.0, 0.0)

        # Calculate bounding box heights
        small_height = max(v[1] for v in small) - min(v[1] for v in small)
        large_height = max(v[1] for v in large) - min(v[1] for v in large)

        assert large_height > small_height

    def test_flip_mirrors_horizontally(self):
        """Flipped horn should be mirrored."""
        base = (0.5, 0.5)
        normal = build_horn(base, 1.0, 0.0, flip=False)
        flipped = build_horn(base, 1.0, 0.0, flip=True)

        # X coordinates should be mirrored around base
        for n, f in zip(normal, flipped):
            expected_x = 2 * base[0] - n[0]
            assert abs(f[0] - expected_x) < 0.001


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
    """Tests for build_halo function."""

    def test_returns_two_circles(self):
        """Should return outer and inner circle arrays."""
        outer, inner = build_halo((0.5, 0.5), 0.2)

        assert isinstance(outer, np.ndarray)
        assert isinstance(inner, np.ndarray)

    def test_outer_larger_than_inner(self):
        """Outer circle should have larger radius."""
        center = (0.5, 0.5)
        outer, inner = build_halo(center, 0.2, thickness=0.05)

        # Calculate average radius of each
        outer_radius = np.mean([math.sqrt((v[0]-center[0])**2 + (v[1]-center[1])**2)
                               for v in outer])
        inner_radius = np.mean([math.sqrt((v[0]-center[0])**2 + (v[1]-center[1])**2)
                               for v in inner])

        assert outer_radius > inner_radius


class TestBuildNeonLines:
    """Tests for build_neon_lines function."""

    def test_returns_list_of_arrays(self):
        """Should return list of line arrays."""
        result = build_neon_lines((0.0, 0.0), (1.0, 1.0), num_lines=3)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_each_line_has_two_points(self):
        """Each line should have start and end points."""
        result = build_neon_lines((0.0, 0.0), (1.0, 1.0))

        for line in result:
            assert len(line) == 2

    def test_zero_length_line_returns_empty(self):
        """Same start and end should return empty list."""
        result = build_neon_lines((0.5, 0.5), (0.5, 0.5))
        assert len(result) == 0

    def test_lines_are_parallel(self):
        """Generated lines should be parallel."""
        result = build_neon_lines((0.0, 0.0), (1.0, 0.0), num_lines=3, spacing=0.1)

        # All lines should be horizontal (same Y for both endpoints)
        for line in result:
            assert abs(line[0][1] - line[1][1]) < 0.001


class TestBuildZigzagLine:
    """Tests for build_zigzag_line function."""

    def test_starts_and_ends_correctly(self):
        """First and last vertices should match start and end."""
        start = (0.1, 0.2)
        end = (0.9, 0.8)
        result = build_zigzag_line(start, end)

        assert abs(result[0][0] - start[0]) < 0.001
        assert abs(result[0][1] - start[1]) < 0.001
        assert abs(result[-1][0] - end[0]) < 0.001
        assert abs(result[-1][1] - end[1]) < 0.001

    def test_correct_vertex_count(self):
        """Should have frequency + 1 vertices."""
        frequency = 8
        result = build_zigzag_line((0.0, 0.0), (1.0, 1.0), frequency=frequency)
        assert len(result) == frequency + 1

    def test_amplitude_affects_deviation(self):
        """Larger amplitude should create larger deviations."""
        small = build_zigzag_line((0.0, 0.0), (1.0, 0.0), amplitude=0.01)
        large = build_zigzag_line((0.0, 0.0), (1.0, 0.0), amplitude=0.1)

        # Calculate max Y deviation for each
        small_dev = max(abs(v[1]) for v in small)
        large_dev = max(abs(v[1]) for v in large)

        assert large_dev > small_dev


class TestBuildStar:
    """Tests for build_star function."""

    def test_returns_numpy_array(self):
        """Should return numpy array."""
        result = build_star((0.5, 0.5), 0.2)
        assert isinstance(result, np.ndarray)

    def test_correct_point_count(self):
        """Should have 2 * points vertices."""
        points = 5
        result = build_star((0.5, 0.5), 0.2, points=points)
        assert len(result) == points * 2

    def test_alternating_radii(self):
        """Vertices should alternate between outer and inner radius."""
        center = (0.5, 0.5)
        outer_r = 0.2
        inner_r = 0.08
        result = build_star(center, outer_r, inner_radius=inner_r, points=5)

        for i, vertex in enumerate(result):
            dist = math.sqrt((vertex[0] - center[0])**2 + (vertex[1] - center[1])**2)
            expected = outer_r if i % 2 == 0 else inner_r
            assert abs(dist - expected) < 0.001


class TestBuildHeart:
    """Tests for build_heart function."""

    def test_returns_numpy_array(self):
        """Should return numpy array."""
        result = build_heart((0.5, 0.5), 1.0)
        assert isinstance(result, np.ndarray)

    def test_has_multiple_vertices(self):
        """Should have multiple vertices for smooth curve."""
        result = build_heart((0.5, 0.5), 1.0)
        assert len(result) >= 20

    def test_centered_around_position(self):
        """Heart should be roughly centered around given position."""
        center = (0.3, 0.7)
        result = build_heart(center, 1.0)

        avg_x = np.mean(result[:, 0])
        avg_y = np.mean(result[:, 1])

        # Should be close to center (with some offset due to heart shape)
        assert abs(avg_x - center[0]) < 0.1
        assert abs(avg_y - center[1]) < 0.2


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

    def test_circle_scales_linearly(self):
        """Circle radius should scale linearly."""
        center = (0.5, 0.5)
        small = build_circle(center, 0.1)
        large = build_circle(center, 0.2)

        small_radius = np.mean([math.sqrt((v[0]-center[0])**2 + (v[1]-center[1])**2)
                               for v in small])
        large_radius = np.mean([math.sqrt((v[0]-center[0])**2 + (v[1]-center[1])**2)
                               for v in large])

        assert abs(large_radius / small_radius - 2.0) < 0.001

    def test_star_scales_with_radius(self):
        """Star size should scale with outer radius."""
        center = (0.5, 0.5)
        small = build_star(center, 0.1)
        large = build_star(center, 0.3)

        small_max = max(math.sqrt((v[0]-center[0])**2 + (v[1]-center[1])**2)
                       for v in small)
        large_max = max(math.sqrt((v[0]-center[0])**2 + (v[1]-center[1])**2)
                       for v in large)

        assert large_max > small_max * 2
