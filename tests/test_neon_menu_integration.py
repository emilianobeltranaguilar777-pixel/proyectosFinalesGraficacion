import math
import sys
import types

import numpy as np
import pytest

from Gesture3D import Gesture3D, Gesture
from neon_menu import MenuState


def _install_fake_cv2():
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
    fake_cv2.resize = lambda img, size, interpolation=None: np.zeros((size[1], size[0], img.shape[2]), dtype=img.dtype)
    fake_cv2.flip = _return_img
    fake_cv2.namedWindow = lambda *_, **__: None
    fake_cv2.resizeWindow = lambda *_, **__: None
    fake_cv2.waitKey = lambda *_: 0
    fake_cv2.VideoCapture = lambda *_, **__: types.SimpleNamespace(
        isOpened=lambda: False, read=lambda: (False, None), set=lambda *a, **k: None
    )

    sys.modules["cv2"] = fake_cv2
    import neon_menu as neon_mod

    neon_mod.cv2 = fake_cv2


@pytest.fixture
def neon_app(monkeypatch):
    """Create a PizarraNeon instance with MediaPipe disabled for fast tests."""

    _install_fake_cv2()
    monkeypatch.setattr(Gesture3D, "_initialize_mediapipe", lambda self: None)
    from main import PizarraNeon

    app = PizarraNeon()
    app.gesture_3d.mediapipe_available = False
    return app


def test_open_hand_does_not_toggle_menu(neon_app):
    app = neon_app
    app.neon_menu.state = MenuState.HIDDEN
    app.neon_menu.animation = 0.0

    app.gesture_3d.current_gesture = Gesture.OPEN_HAND
    app._actualizar_neon_menu(0.1)

    assert app.neon_menu.state == MenuState.HIDDEN
    assert app.neon_menu.animation == 0.0


def test_victory_toggle_opens_and_closes(neon_app):
    app = neon_app
    app.neon_menu.state = MenuState.HIDDEN
    app.neon_menu.animation = 0.0
    app.gesture_3d.last_victory_time = -1.0

    first_time = 0.0
    app.gesture_3d.handle_gestures(Gesture.VICTORY, (100, 120), first_time)
    app._actualizar_neon_menu(0.05)

    assert app.neon_menu.is_visible()

    app.gesture_3d.handle_gestures(Gesture.VICTORY, (100, 120), first_time + 1.0)
    app._actualizar_neon_menu(0.05)

    assert app.neon_menu.state in (MenuState.CLOSING, MenuState.HIDDEN)


def test_rotation_locked_when_menu_visible(neon_app):
    app = neon_app
    app.neon_menu.state = MenuState.VISIBLE
    app.neon_menu.animation = 1.0

    app.gesture_3d.selected_figure = {
        "type": "circle",
        "position": (100, 100),
        "size": 50,
        "color": (255, 0, 0),
        "rotation": 1.0,
        "selection_color": (255, 255, 80),
        "creation_time": 0.0,
        "id": 1,
    }

    app.gesture_3d.rotation_enabled = True
    app.gesture_3d.external_menu_active = True
    app.gesture_3d.handle_gestures(Gesture.OPEN_HAND, (120, 120), 1.0)

    assert math.isclose(app.gesture_3d.selected_figure["rotation"], 1.0)


def test_rotation_resumes_after_menu_close(monkeypatch):
    _install_fake_cv2()
    monkeypatch.setattr(Gesture3D, "_initialize_mediapipe", lambda self: None)
    g3d = Gesture3D(640, 480, use_external_menu=True)
    g3d.mediapipe_available = False
    g3d.selected_figure = {
        "type": "circle",
        "position": (100, 100),
        "size": 50,
        "color": (255, 0, 0),
        "rotation": 1.0,
        "selection_color": (255, 255, 80),
        "creation_time": 0.0,
        "id": 1,
    }

    g3d.external_menu_active = True
    g3d.handle_gestures(Gesture.OPEN_HAND, (120, 120), 1.0)
    assert math.isclose(g3d.selected_figure["rotation"], 1.0)

    g3d.external_menu_active = False
    g3d.rotation_enabled = True
    g3d.handle_gestures(Gesture.OPEN_HAND, (120, 120), 1.2)

    assert g3d.selected_figure["rotation"] > 1.0
