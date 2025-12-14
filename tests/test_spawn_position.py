from tests.conftest import _install_fake_cv2


def _fake_app(monkeypatch):
    _install_fake_cv2()
    from Gesture3D import Gesture3D
    monkeypatch.setattr(Gesture3D, "_initialize_mediapipe", lambda self: None)
    from main import PizarraNeon

    app = PizarraNeon()
    app.gesture_3d.mediapipe_available = False
    return app


def test_spawn_clamped_inside_frame(monkeypatch):
    app = _fake_app(monkeypatch)
    app.gesture_3d.last_pinch_position = (-50, app.alto + 100)

    pos = app._posicion_segura_creacion()

    margen = 80
    assert margen <= pos[0] <= app.ancho - margen
    assert margen <= pos[1] <= app.alto - margen


def test_spawn_prefers_valid_pinch(monkeypatch):
    app = _fake_app(monkeypatch)
    app.gesture_3d.last_pinch_position = (150, 200)

    pos = app._posicion_segura_creacion()

    assert pos == (150, 200)
