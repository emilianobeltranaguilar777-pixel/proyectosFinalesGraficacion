import math
import sys
import time
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
    fake_cv2.resize = lambda img, size: np.zeros((size[1], size[0], img.shape[2]), dtype=img.dtype)
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
    app.tiempo_apertura_requerido = 0.1
    return app


def test_neon_menu_opens_and_closes(neon_app):
    app = neon_app
    app.neon_menu.state = MenuState.HIDDEN
    app.neon_menu.animation = 0.0

    app.gesture_3d.current_gesture = Gesture.OPEN_HAND
    for _ in range(3):
        app._actualizar_neon_menu(0.05)

    assert app.neon_menu.is_visible()
    assert app.neon_menu.state in (MenuState.OPENING, MenuState.VISIBLE)

    app.gesture_3d.current_gesture = Gesture.FIST
    app._actualizar_neon_menu(0.3)
    app._actualizar_neon_menu(0.3)

    assert app.neon_menu.state == MenuState.HIDDEN
    assert app.neon_menu.animation == 0.0


def test_rotation_locked_while_menu_visible(neon_app):
    app = neon_app
    app.neon_menu.state = MenuState.VISIBLE
    app.neon_menu.animation = 1.0

    frame = np.zeros((app.alto, app.ancho, 3), dtype=np.uint8)
    app.gesture_3d.rotation_enabled = True

    app.modo_figuras_gestos(frame)

    assert not app.gesture_3d.rotation_enabled
    assert app.gesture_3d.external_menu_active


def test_pinch_selection_triggers_once(neon_app):
    app = neon_app
    menu = app.neon_menu
    menu.state = MenuState.VISIBLE
    menu.animation = 1.0

    sector_angle = (2 * math.pi) / len(menu.buttons)
    angle = menu.start_angle + sector_angle * 0.5
    target_pos = menu._polar_to_cartesian(angle, menu.radius * 0.9)
    app.ultima_pos_cursor = target_pos

    callbacks = []
    menu.buttons[0].on_select = lambda btn: callbacks.append(btn.label)

    app.gesture_3d.current_gesture = Gesture.NONE
    app.gesture_3d.pinch_active = True
    app.prev_pinch_activo = False
    app._actualizar_neon_menu(0.016)

    app._actualizar_neon_menu(0.016)
    app.gesture_3d.pinch_active = False
    app._actualizar_neon_menu(0.016)
    app.gesture_3d.pinch_active = True
    app._actualizar_neon_menu(0.016)

    assert callbacks == ["circle", "circle"]


def test_no_rotation_when_external_menu_active(monkeypatch):
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
    g3d.rotation_enabled = True

    g3d.handle_gestures(Gesture.OPEN_HAND, (120, 120), time.perf_counter())

    assert math.isclose(g3d.selected_figure["rotation"], 1.0)
    assert g3d.pinch_active is False
