"""Fixtures compartidas para pytest."""
import math
from pathlib import Path
import sys
import types

import pytest
import numpy as np


def _install_fake_cv2():
    """Instala un stub mínimo de cv2 solo si la librería real no está disponible."""

    try:  # Preferir la implementación real para no interferir con ColorPainter
        import cv2 as real_cv2

        return real_cv2
    except Exception:  # pragma: no cover - solo en entornos sin OpenCV
        fake_cv2 = types.SimpleNamespace(
            LINE_AA=1,
            FONT_HERSHEY_SIMPLEX=0,
            WINDOW_NORMAL=0,
            COLOR_BGR2RGB=0,
            COLOR_HSV2BGR=1,
            COLOR_BGR2HSV=2,
            MORPH_OPEN=0,
            MORPH_CLOSE=1,
            RETR_EXTERNAL=0,
            CHAIN_APPROX_SIMPLE=0,
        )

        def _return_img(img, *_, **__):
            return img

        def _cvt_color(img, code, *_args, **_kwargs):
            if code == fake_cv2.COLOR_HSV2BGR:
                hsv = img.astype(np.float32)
                h = hsv[..., 0] / 179.0
                s = hsv[..., 1] / 255.0
                v = hsv[..., 2] / 255.0
                i = (h * 6.0).astype(int)
                f = (h * 6.0) - i
                p = v * (1.0 - s)
                q = v * (1.0 - f * s)
                t = v * (1.0 - (1.0 - f) * s)
                i_mod = i % 6
                conditions = [
                    (i_mod == 0, np.stack([v, t, p], axis=-1)),
                    (i_mod == 1, np.stack([q, v, p], axis=-1)),
                    (i_mod == 2, np.stack([p, v, t], axis=-1)),
                    (i_mod == 3, np.stack([p, q, v], axis=-1)),
                    (i_mod == 4, np.stack([t, p, v], axis=-1)),
                    (i_mod == 5, np.stack([v, p, q], axis=-1)),
                ]
                bgr = np.zeros_like(hsv, dtype=np.float32)
                for cond, val in conditions:
                    bgr[..., 0] = np.where(cond, val[..., 2], bgr[..., 0])
                    bgr[..., 1] = np.where(cond, val[..., 1], bgr[..., 1])
                    bgr[..., 2] = np.where(cond, val[..., 0], bgr[..., 2])
                return np.clip(bgr * 255, 0, 255).astype(np.uint8)
            elif code == fake_cv2.COLOR_BGR2HSV:
                bgr = img.astype(np.float32) / 255.0
                b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
                c_max = np.maximum(np.maximum(r, g), b)
                c_min = np.minimum(np.minimum(r, g), b)
                delta = c_max - c_min

                h = np.zeros_like(c_max)
                mask = delta != 0
                h[mask & (c_max == r)] = (60 * ((g - b) / delta) + 360)[mask & (c_max == r)]
                h[mask & (c_max == g)] = (60 * ((b - r) / delta) + 120)[mask & (c_max == g)]
                h[mask & (c_max == b)] = (60 * ((r - g) / delta) + 240)[mask & (c_max == b)]
                h = (h % 360) / 2

                s = np.where(c_max == 0, 0, delta / c_max)
                v = c_max
                hsv = np.stack([h, s * 255.0, v * 255.0], axis=-1)
                return hsv.astype(np.uint8)
            return img

        def _in_range(src, lower, upper):
            lower = np.array(lower, dtype=src.dtype)
            upper = np.array(upper, dtype=src.dtype)
            mask = np.all((src >= lower) & (src <= upper), axis=-1)
            return mask.astype(np.uint8) * 255

        def _circle(img, center, radius, color, thickness=-1, lineType=None):
            yy, xx = np.ogrid[: img.shape[0], : img.shape[1]]
            dist = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
            if thickness == -1:
                mask = dist <= radius ** 2
                img[mask] = color
            else:
                inner = (radius - thickness) ** 2
                mask = (dist <= radius ** 2) & (dist >= inner)
                img[mask] = color
            return img

        def _find_contours(mask, *_args, **_kwargs):
            mask_bool = mask.astype(bool)
            if not mask_bool.any():
                return [], None

            visited = np.zeros_like(mask_bool, dtype=bool)
            contours = []
            height, width = mask_bool.shape
            neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

            for y, x in np.argwhere(mask_bool):
                if visited[y, x]:
                    continue
                stack = [(y, x)]
                component = []
                visited[y, x] = True

                while stack:
                    cy, cx = stack.pop()
                    component.append([cx, cy])

                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width and mask_bool[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                contours.append(np.array(component, dtype=np.int32))

            return contours, None

        def _contour_area(contour):
            return float(contour.shape[0])

        def _moments(contour):
            area = _contour_area(contour)
            if area == 0:
                return {"m00": 0, "m10": 0, "m01": 0}
            cx = contour[:, 0].mean()
            cy = contour[:, 1].mean()
            return {"m00": area, "m10": area * cx, "m01": area * cy}

        fake_cv2.putText = _return_img
        fake_cv2.line = _return_img
        fake_cv2.rectangle = _return_img
        fake_cv2.polylines = _return_img
        fake_cv2.ellipse = _return_img
        fake_cv2.cvtColor = _cvt_color
        fake_cv2.GaussianBlur = lambda img, *_args, **_kwargs: img
        fake_cv2.addWeighted = lambda src1, alpha, src2, beta, gamma, dst=None: src1 if dst is None else dst
        fake_cv2.resize = lambda img, size, interpolation=None: np.zeros((size[1], size[0], img.shape[2]), dtype=img.dtype)
        fake_cv2.flip = _return_img
        fake_cv2.namedWindow = lambda *_, **__: None
        fake_cv2.resizeWindow = lambda *_, **__: None
        fake_cv2.waitKey = lambda *_: 0
        fake_cv2.getTickCount = lambda: 0
        fake_cv2.getTickFrequency = lambda: 1.0
        fake_cv2.VideoCapture = lambda *_, **__: types.SimpleNamespace(
            isOpened=lambda: False, read=lambda: (False, None), set=lambda *a, **k: None
        )
        fake_cv2.inRange = _in_range
        fake_cv2.morphologyEx = lambda img, *_args, **_kwargs: img
        fake_cv2.findContours = _find_contours
        fake_cv2.contourArea = _contour_area
        fake_cv2.moments = _moments
        fake_cv2.circle = _circle

        sys.modules["cv2"] = fake_cv2
        return fake_cv2


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
