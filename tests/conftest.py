"""Fixtures compartidas para pytest."""
import sys
import math
from pathlib import Path

import pytest
import numpy as np

# Agregar el directorio raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_square_points():
    """Puntos de un cuadrado centrado en (100, 100) con lado 20."""
    return np.array([
        [90, 90],
        [110, 90],
        [110, 110],
        [90, 110]
    ], dtype=np.float64)


@pytest.fixture
def sample_triangle_points():
    """Puntos de un triangulo centrado en (100, 100)."""
    return np.array([
        [100, 80],
        [80, 120],
        [120, 120]
    ], dtype=np.float64)


@pytest.fixture
def center_point():
    """Centro de rotacion estandar."""
    return (100, 100)


@pytest.fixture
def mock_gesture3d():
    """Crea una instancia de Gesture3D sin MediaPipe."""
    from Gesture3D import Gesture3D

    g3d = Gesture3D(640, 480)
    # Deshabilitar MediaPipe para tests
    g3d.mediapipe_available = False
    return g3d


@pytest.fixture
def mock_figure():
    """Figura de prueba."""
    return {
        'type': 'square',
        'position': (100, 100),
        'size': 50,
        'color': (255, 0, 0),
        'rotation': 0.0,
        'selection_color': (255, 255, 80),
        'creation_time': 0,
        'id': 0
    }
