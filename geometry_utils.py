"""Utilidades geometricas para transformaciones 2D."""
import numpy as np


def rotate_points(points: np.ndarray, center: tuple, angle_rad: float) -> np.ndarray:
    """
    Rota puntos 2D alrededor de un centro.

    Args:
        points: Array de puntos con shape (N, 2)
        center: Tupla (cx, cy) del centro de rotacion
        angle_rad: Angulo en radianes (positivo = antihorario)

    Returns:
        Array de puntos rotados con dtype np.int32
    """
    if len(points) == 0:
        return np.array([], dtype=np.int32)

    points = np.asarray(points, dtype=np.float64)
    cx, cy = center

    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    translated = points - np.array([cx, cy])

    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])

    rotated = translated @ rotation_matrix.T

    result = rotated + np.array([cx, cy])

    return result.astype(np.int32)
