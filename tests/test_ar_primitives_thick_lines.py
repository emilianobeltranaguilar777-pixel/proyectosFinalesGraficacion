"""
Tests for Thick Line Primitives in AR Filter

These tests validate that the thick line primitives:
- Generate triangle-based geometry
- Have correct vertex counts (multiples of 6 for triangles)
- Have non-zero area
- Work correctly with various inputs
"""

import pytest
import math
import numpy as np

from ar_filter.primitives import (
    build_thick_line,
    build_thick_line_strip,
    build_thick_circle,
    build_thick_star,
    build_thick_polygon,
    build_filled_quad,
)


class TestBuildThickLine:
    """Tests for build_thick_line function."""

    def test_returns_6_vertices(self):
        """A single thick line should return 6 vertices (2 triangles)."""
        result = build_thick_line((0.0, 0.0), (1.0, 0.0), thickness=0.01)

        assert len(result) == 6, \
            f"Expected 6 vertices (2 triangles), got {len(result)}"

    def test_returns_float32(self):
        """Result should be float32 numpy array."""
        result = build_thick_line((0.0, 0.0), (1.0, 0.0))

        assert result.dtype == np.float32

    def test_degenerate_line_returns_empty(self):
        """Same start and end should return empty array."""
        result = build_thick_line((0.5, 0.5), (0.5, 0.5))

        assert len(result) == 0

    def test_has_nonzero_area(self):
        """Thick line should have non-zero area."""
        result = build_thick_line((0.0, 0.0), (1.0, 0.0), thickness=0.1)

        # Calculate bounding box area
        min_x = np.min(result[:, 0])
        max_x = np.max(result[:, 0])
        min_y = np.min(result[:, 1])
        max_y = np.max(result[:, 1])

        width = max_x - min_x
        height = max_y - min_y

        assert width > 0, "Thick line should have non-zero width"
        assert height > 0, "Thick line should have non-zero height (thickness)"

    def test_thickness_affects_height(self):
        """Larger thickness should produce taller line."""
        thin = build_thick_line((0.0, 0.0), (1.0, 0.0), thickness=0.01)
        thick = build_thick_line((0.0, 0.0), (1.0, 0.0), thickness=0.1)

        thin_height = np.max(thin[:, 1]) - np.min(thin[:, 1])
        thick_height = np.max(thick[:, 1]) - np.min(thick[:, 1])

        assert thick_height > thin_height, \
            "Larger thickness should produce larger height"


class TestBuildThickLineStrip:
    """Tests for build_thick_line_strip function."""

    def test_two_points_returns_6_vertices(self):
        """Two points should return one segment (6 vertices)."""
        points = [(0.0, 0.0), (1.0, 0.0)]
        result = build_thick_line_strip(points)

        assert len(result) == 6

    def test_three_points_returns_12_vertices(self):
        """Three points should return two segments (12 vertices)."""
        points = [(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]
        result = build_thick_line_strip(points)

        assert len(result) == 12

    def test_empty_input_returns_empty(self):
        """Empty input should return empty array."""
        result = build_thick_line_strip([])

        assert len(result) == 0

    def test_single_point_returns_empty(self):
        """Single point should return empty array."""
        result = build_thick_line_strip([(0.5, 0.5)])

        assert len(result) == 0

    def test_vertex_count_is_multiple_of_6(self):
        """Vertex count should always be multiple of 6."""
        points = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.0), (0.75, 0.25), (1.0, 0.0)]
        result = build_thick_line_strip(points)

        assert len(result) % 6 == 0, \
            "Vertex count must be multiple of 6 (2 triangles per segment)"


class TestBuildThickCircle:
    """Tests for build_thick_circle function."""

    def test_default_segments(self):
        """Default 32 segments should produce 192 vertices."""
        result = build_thick_circle((0.5, 0.5), 0.1)

        # 32 segments * 6 vertices per segment = 192
        assert len(result) == 192

    def test_custom_segments(self):
        """Custom segment count should work."""
        result = build_thick_circle((0.5, 0.5), 0.1, segments=16)

        assert len(result) == 96  # 16 * 6

    def test_is_ring_shaped(self):
        """Circle should have inner and outer radius."""
        center = (0.5, 0.5)
        radius = 0.2
        thickness = 0.02

        result = build_thick_circle(center, radius, thickness=thickness)

        # Calculate distances from center
        distances = np.sqrt((result[:, 0] - center[0])**2 +
                           (result[:, 1] - center[1])**2)

        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Should have inner and outer edges
        assert min_dist < radius, "Should have inner edge"
        assert max_dist > radius, "Should have outer edge"


class TestBuildThickStar:
    """Tests for build_thick_star function."""

    def test_has_correct_structure(self):
        """Star should return triangle vertices."""
        result = build_thick_star((0.5, 0.5), 0.2)

        assert len(result) > 0
        assert len(result) % 6 == 0, \
            "Star vertex count should be multiple of 6"

    def test_custom_points(self):
        """Custom number of points should work."""
        star5 = build_thick_star((0.5, 0.5), 0.2, points=5)
        star6 = build_thick_star((0.5, 0.5), 0.2, points=6)

        # More points = more segments = more vertices
        assert len(star6) > len(star5)

    def test_inner_radius(self):
        """Custom inner radius should affect shape."""
        thin = build_thick_star((0.5, 0.5), 0.2, inner_radius=0.15)
        fat = build_thick_star((0.5, 0.5), 0.2, inner_radius=0.05)

        # Both should have vertices
        assert len(thin) > 0
        assert len(fat) > 0


class TestBuildThickPolygon:
    """Tests for build_thick_polygon function."""

    def test_triangle_polygon(self):
        """Triangle polygon should return vertices."""
        points = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        result = build_thick_polygon(points, closed=True)

        assert len(result) > 0
        assert len(result) % 6 == 0

    def test_closed_polygon_has_more_vertices(self):
        """Closed polygon should have one more segment than open."""
        points = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]

        closed = build_thick_polygon(points, closed=True)
        opened = build_thick_polygon(points, closed=False)

        # Closed has extra segment connecting last to first
        assert len(closed) > len(opened)


class TestBuildFilledQuad:
    """Tests for build_filled_quad function."""

    def test_returns_6_vertices(self):
        """Quad should return 6 vertices (2 triangles)."""
        result = build_filled_quad(
            (0.0, 0.0), (1.0, 0.0),
            (0.0, 1.0), (1.0, 1.0)
        )

        assert len(result) == 6

    def test_covers_expected_area(self):
        """Quad should cover the expected area."""
        result = build_filled_quad(
            (0.0, 0.0), (1.0, 0.0),
            (0.0, 1.0), (1.0, 1.0)
        )

        min_x = np.min(result[:, 0])
        max_x = np.max(result[:, 0])
        min_y = np.min(result[:, 1])
        max_y = np.max(result[:, 1])

        assert min_x == 0.0
        assert max_x == 1.0
        assert min_y == 0.0
        assert max_y == 1.0


class TestVertexValidity:
    """Tests for vertex data validity."""

    def test_no_nan_values(self):
        """Thick primitives should not contain NaN."""
        primitives = [
            build_thick_line((0.0, 0.0), (1.0, 1.0)),
            build_thick_circle((0.5, 0.5), 0.2),
            build_thick_star((0.5, 0.5), 0.1),
        ]

        for prim in primitives:
            assert not np.any(np.isnan(prim)), \
                "Primitives should not contain NaN"

    def test_no_infinite_values(self):
        """Thick primitives should not contain infinite values."""
        primitives = [
            build_thick_line((0.0, 0.0), (1.0, 1.0)),
            build_thick_circle((0.5, 0.5), 0.2),
            build_thick_star((0.5, 0.5), 0.1),
        ]

        for prim in primitives:
            assert not np.any(np.isinf(prim)), \
                "Primitives should not contain infinite values"

    def test_all_2d_vertices(self):
        """All vertices should be 2D."""
        primitives = [
            build_thick_line((0.0, 0.0), (1.0, 1.0)),
            build_thick_line_strip([(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)]),
            build_thick_circle((0.5, 0.5), 0.2),
            build_thick_star((0.5, 0.5), 0.1),
        ]

        for prim in primitives:
            if len(prim) > 0:
                assert prim.ndim == 2, "Vertices should be 2D array"
                assert prim.shape[1] == 2, "Each vertex should have 2 components"


class TestTriangleArea:
    """Tests for non-zero triangle area."""

    def _triangle_area(self, v0, v1, v2):
        """Calculate area of a triangle from 3 vertices."""
        return abs((v1[0] - v0[0]) * (v2[1] - v0[1]) -
                   (v2[0] - v0[0]) * (v1[1] - v0[1])) / 2.0

    def test_thick_line_triangles_have_area(self):
        """Each triangle in thick line should have non-zero area."""
        result = build_thick_line((0.0, 0.0), (1.0, 0.0), thickness=0.1)

        # Check each triangle (every 3 vertices)
        for i in range(0, len(result), 3):
            area = self._triangle_area(result[i], result[i+1], result[i+2])
            assert area > 0, f"Triangle {i//3} has zero area!"

    def test_thick_circle_triangles_have_area(self):
        """Each triangle in thick circle should have non-zero area."""
        result = build_thick_circle((0.5, 0.5), 0.2, thickness=0.02, segments=8)

        # Check each triangle
        for i in range(0, len(result), 3):
            area = self._triangle_area(result[i], result[i+1], result[i+2])
            assert area > 0, f"Triangle {i//3} has zero area!"
