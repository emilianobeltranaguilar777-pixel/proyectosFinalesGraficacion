"""Fixtures compartidas para pytest."""
import math
from pathlib import Path
import sys
import types

import sys
import types
from pathlib import Path

import pytest
import numpy as np


def _install_fake_cv2():
    """Instala un stub mínimo de cv2 para entornos headless."""

    fake_cv2 = types.SimpleNamespace(
        LINE_AA=1,
        FONT_HERSHEY_SIMPLEX=0,
        WINDOW_NORMAL=0,
        COLOR_BGR2RGB=0,
    )

    def _return_img(img, *_, **__):
        return img

    fake_cv2.putText = _return_img
    fake_cv2.line = _return_img
    fake_cv2.circle = _return_img
    fake_cv2.rectangle = _return_img
    fake_cv2.polylines = _return_img
    fake_cv2.cvtColor = lambda img, *_args, **_kwargs: img
    fake_cv2.GaussianBlur = lambda img, *_args, **_kwargs: img
    fake_cv2.addWeighted = lambda src1, alpha, src2, beta, gamma, dst=None: src1 if dst is None else dst
    fake_cv2.resize = lambda img, size: np.zeros((size[1], size[0], img.shape[2]), dtype=img.dtype)
    fake_cv2.flip = _return_img
    fake_cv2.namedWindow = lambda *_, **__: None
    fake_cv2.resizeWindow = lambda *_, **__: None
    fake_cv2.waitKey = lambda *_: 0
    fake_cv2.VideoCapture = lambda *_, **__: types.SimpleNamespace(
        isOpened=lambda: False, read=lambda: (False, None), set=lambda *a, **k: None
    )

    sys.modules["cv2"] = fake_cv2


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
def mock_gesture3d(monkeypatch):
    """Crea una instancia de Gesture3D sin MediaPipe."""

    _install_fake_cv2()
    from Gesture3D import Gesture3D

    monkeypatch.setattr(Gesture3D, "_initialize_mediapipe", lambda self: None)
    g3d = Gesture3D(640, 480)
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
