"""
Tests for AR filter primitives module.

These tests verify geometry generation without any OpenGL dependencies.
"""

import pytest
import numpy as np
import math
from ar_filter.primitives import (
    build_sphere,
    build_quad,
    build_icosphere,
    sphere_vertex_count,
    sphere_triangle_count,
    icosphere_vertex_count,
)


class TestBuildSphere:
    """Tests for build_sphere function."""

    def test_sphere_returns_tuple(self):
        """Should return tuple of (vertices, normals, indices)."""
        result = build_sphere()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_sphere_vertex_count(self):
        """Should have correct number of vertices."""
        lat, lon = 8, 12
        vertices, normals, indices = build_sphere(lat_segments=lat, lon_segments=lon)

        expected = (lat + 1) * (lon + 1)
        assert len(vertices) == expected
        assert len(normals) == expected

    def test_sphere_data_types(self):
        """Should return correct numpy dtypes."""
        vertices, normals, indices = build_sphere()

        assert vertices.dtype == np.float32
        assert normals.dtype == np.float32
        assert indices.dtype == np.uint32

    def test_sphere_vertex_shape(self):
        """Vertices should be Nx3 arrays."""
        vertices, normals, indices = build_sphere()

        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert normals.shape == vertices.shape

    def test_sphere_indices_shape(self):
        """Indices should be Mx3 arrays (triangles)."""
        vertices, normals, indices = build_sphere()

        assert indices.ndim == 2
        assert indices.shape[1] == 3

    def test_sphere_radius(self):
        """Vertices should be at specified radius."""
        radius = 2.5
        vertices, normals, indices = build_sphere(radius=radius)

        # Check that vertices are at approximately the correct radius
        distances = np.linalg.norm(vertices, axis=1)
        assert np.allclose(distances, radius, atol=0.001)

    def test_sphere_normals_unit_length(self):
        """Normals should be unit vectors."""
        vertices, normals, indices = build_sphere()

        lengths = np.linalg.norm(normals, axis=1)
        assert np.allclose(lengths, 1.0, atol=0.001)

    def test_sphere_triangle_count(self):
        """Should have correct number of triangles."""
        lat, lon = 8, 12
        vertices, normals, indices = build_sphere(lat_segments=lat, lon_segments=lon)

        expected_triangles = lat * lon * 2
        assert len(indices) == expected_triangles

    def test_sphere_valid_indices(self):
        """All indices should reference valid vertices."""
        vertices, normals, indices = build_sphere()

        max_index = indices.max()
        assert max_index < len(vertices)


class TestBuildQuad:
    """Tests for build_quad function."""

    def test_quad_returns_tuple(self):
        """Should return tuple of (vertices, normals, indices)."""
        result = build_quad()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_quad_vertex_count(self):
        """Should have exactly 4 vertices."""
        vertices, normals, indices = build_quad()
        assert len(vertices) == 4
        assert len(normals) == 4

    def test_quad_triangle_count(self):
        """Should have exactly 2 triangles."""
        vertices, normals, indices = build_quad()
        assert len(indices) == 2

    def test_quad_dimensions(self):
        """Quad should have correct width and height."""
        width, height = 3.0, 2.0
        vertices, normals, indices = build_quad(width=width, height=height)

        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]

        actual_width = x_coords.max() - x_coords.min()
        actual_height = y_coords.max() - y_coords.min()

        assert abs(actual_width - width) < 0.001
        assert abs(actual_height - height) < 0.001

    def test_quad_center(self):
        """Quad should be centered at specified position."""
        center = (1.0, 2.0, 3.0)
        vertices, normals, indices = build_quad(center=center)

        centroid = vertices.mean(axis=0)
        assert np.allclose(centroid, center, atol=0.001)

    def test_quad_normals_face_forward(self):
        """Normals should all point in +Z direction."""
        vertices, normals, indices = build_quad()

        expected_normal = np.array([0.0, 0.0, 1.0])
        for normal in normals:
            assert np.allclose(normal, expected_normal)

    def test_quad_data_types(self):
        """Should return correct numpy dtypes."""
        vertices, normals, indices = build_quad()

        assert vertices.dtype == np.float32
        assert normals.dtype == np.float32
        assert indices.dtype == np.uint32


class TestBuildIcosphere:
    """Tests for build_icosphere function."""

    def test_icosphere_returns_tuple(self):
        """Should return tuple of (vertices, normals, indices)."""
        result = build_icosphere()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_icosphere_base_vertex_count(self):
        """Base icosahedron should have 12 vertices."""
        vertices, normals, indices = build_icosphere(subdivisions=0)
        assert len(vertices) == 12

    def test_icosphere_subdivision_increases_vertices(self):
        """Each subdivision should increase vertex count."""
        v0, _, _ = build_icosphere(subdivisions=0)
        v1, _, _ = build_icosphere(subdivisions=1)
        v2, _, _ = build_icosphere(subdivisions=2)

        assert len(v1) > len(v0)
        assert len(v2) > len(v1)

    def test_icosphere_radius(self):
        """Vertices should be at specified radius."""
        radius = 1.5
        vertices, normals, indices = build_icosphere(radius=radius, subdivisions=1)

        distances = np.linalg.norm(vertices, axis=1)
        assert np.allclose(distances, radius, atol=0.001)

    def test_icosphere_normals_unit_length(self):
        """Normals should be unit vectors."""
        vertices, normals, indices = build_icosphere(subdivisions=1)

        lengths = np.linalg.norm(normals, axis=1)
        assert np.allclose(lengths, 1.0, atol=0.001)

    def test_icosphere_data_types(self):
        """Should return correct numpy dtypes."""
        vertices, normals, indices = build_icosphere()

        assert vertices.dtype == np.float32
        assert normals.dtype == np.float32
        assert indices.dtype == np.uint32

    def test_icosphere_valid_indices(self):
        """All indices should reference valid vertices."""
        vertices, normals, indices = build_icosphere(subdivisions=1)

        max_index = indices.max()
        assert max_index < len(vertices)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_sphere_vertex_count_function(self):
        """sphere_vertex_count should match actual sphere."""
        lat, lon = 10, 16
        vertices, _, _ = build_sphere(lat_segments=lat, lon_segments=lon)

        expected = sphere_vertex_count(lat, lon)
        assert len(vertices) == expected

    def test_sphere_triangle_count_function(self):
        """sphere_triangle_count should match actual sphere."""
        lat, lon = 10, 16
        _, _, indices = build_sphere(lat_segments=lat, lon_segments=lon)

        expected = sphere_triangle_count(lat, lon)
        assert len(indices) == expected

    def test_icosphere_vertex_count_approximate(self):
        """icosphere_vertex_count should be reasonable approximation."""
        for subdivisions in range(3):
            vertices, _, _ = build_icosphere(subdivisions=subdivisions)
            estimated = icosphere_vertex_count(subdivisions)

            # Should be within 10% of actual
            ratio = len(vertices) / estimated
            assert 0.9 < ratio < 1.1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_sphere_minimum_segments(self):
        """Sphere with minimum segments should work."""
        vertices, normals, indices = build_sphere(lat_segments=2, lon_segments=3)

        assert len(vertices) > 0
        assert len(indices) > 0

    def test_quad_zero_dimensions(self):
        """Zero-dimension quad should create degenerate geometry."""
        vertices, normals, indices = build_quad(width=0.0, height=0.0)

        # All vertices should be at the same point
        assert len(vertices) == 4

    def test_icosphere_zero_subdivisions(self):
        """Icosphere with 0 subdivisions should be valid icosahedron."""
        vertices, normals, indices = build_icosphere(subdivisions=0)

        assert len(vertices) == 12
        assert len(indices) == 20  # Icosahedron has 20 faces

    def test_sphere_small_radius(self):
        """Very small radius should work correctly."""
        radius = 0.001
        vertices, normals, indices = build_sphere(radius=radius)

        distances = np.linalg.norm(vertices, axis=1)
        assert np.allclose(distances, radius, atol=0.0001)
