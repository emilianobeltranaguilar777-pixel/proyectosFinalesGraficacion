"""Tests para geometry_utils.py"""
import math

import numpy as np
import pytest

from geometry_utils import rotate_points


class TestRotatePoints:
    """Tests para la funcion rotate_points."""

    def test_rotate_points_90_deg(self, sample_square_points, center_point):
        """Test rotacion de 90 grados en sentido antihorario."""
        angle = math.pi / 2  # 90 grados

        rotated = rotate_points(sample_square_points, center_point, angle)

        # Cuadrado rotado 90 grados antihorario alrededor de (100, 100)
        # (90, 90) -> (110, 90)
        # (110, 90) -> (110, 110)
        # (110, 110) -> (90, 110)
        # (90, 110) -> (90, 90)
        expected = np.array([
            [110, 90],
            [110, 110],
            [90, 110],
            [90, 90]
        ], dtype=np.int32)

        np.testing.assert_array_equal(rotated, expected)

    def test_rotate_points_180_deg(self, sample_square_points, center_point):
        """Test rotacion de 180 grados."""
        angle = math.pi  # 180 grados

        rotated = rotate_points(sample_square_points, center_point, angle)

        # Cuadrado rotado 180 grados (opuesto)
        expected = np.array([
            [110, 110],
            [90, 110],
            [90, 90],
            [110, 90]
        ], dtype=np.int32)

        np.testing.assert_array_equal(rotated, expected)

    def test_rotate_points_360_deg(self, sample_square_points, center_point):
        """Test rotacion de 360 grados (debe volver al original)."""
        angle = 2 * math.pi  # 360 grados

        rotated = rotate_points(sample_square_points, center_point, angle)

        # Debe ser igual al original (con tolerancia por flotantes)
        np.testing.assert_array_almost_equal(
            rotated,
            sample_square_points.astype(np.int32),
            decimal=0
        )

    def test_rotate_points_zero_angle(self, sample_square_points, center_point):
        """Test con angulo cero (sin rotacion)."""
        angle = 0.0

        rotated = rotate_points(sample_square_points, center_point, angle)

        expected = sample_square_points.astype(np.int32)
        np.testing.assert_array_equal(rotated, expected)

    def test_rotate_points_returns_int32(self, sample_square_points, center_point):
        """Test que el resultado sea dtype int32."""
        rotated = rotate_points(sample_square_points, center_point, math.pi / 4)

        assert rotated.dtype == np.int32

    def test_rotate_points_empty_array(self, center_point):
        """Test con array vacio."""
        empty = np.array([], dtype=np.float64)

        rotated = rotate_points(empty, center_point, math.pi / 2)

        assert len(rotated) == 0
        assert rotated.dtype == np.int32

    def test_rotate_triangle_45_deg(self, sample_triangle_points, center_point):
        """Test rotacion de triangulo a 45 grados."""
        angle = math.pi / 4  # 45 grados

        rotated = rotate_points(sample_triangle_points, center_point, angle)

        # Verificar que el resultado tenga la forma correcta
        assert rotated.shape == (3, 2)
        assert rotated.dtype == np.int32

        # Verificar que la rotacion preserva la distancia de cada punto al centro
        cx, cy = center_point
        for orig, rot in zip(sample_triangle_points, rotated):
            dist_orig = math.hypot(orig[0] - cx, orig[1] - cy)
            dist_rot = math.hypot(rot[0] - cx, rot[1] - cy)
            assert abs(dist_orig - dist_rot) <= 1  # Tolerancia de 1 pixel

    def test_rotate_points_preserves_distances(self, sample_square_points, center_point):
        """Test que la rotacion preserva distancias al centro."""
        angle = math.pi / 3  # 60 grados

        rotated = rotate_points(sample_square_points, center_point, angle)

        # Calcular distancias al centro antes y despues
        cx, cy = center_point
        original_distances = [
            math.hypot(p[0] - cx, p[1] - cy)
            for p in sample_square_points
        ]
        rotated_distances = [
            math.hypot(p[0] - cx, p[1] - cy)
            for p in rotated
        ]

        # Las distancias deben ser iguales (con tolerancia)
        np.testing.assert_array_almost_equal(
            original_distances,
            rotated_distances,
            decimal=0
        )
