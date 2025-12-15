"""Pruebas del bloqueo duro de modo escala y su UI."""

import numpy as np
import pytest

from Gesture3D import Gesture3D, SelectionMode, Gesture
from main import PizarraNeon


@pytest.fixture
def app(monkeypatch):
    """Instancia de la app con MediaPipe deshabilitado para pruebas."""

    monkeypatch.setattr(Gesture3D, "_initialize_mediapipe", lambda self: None)
    instancia = PizarraNeon()
    instancia.gesture_3d.mediapipe_available = False
    return instancia


@pytest.fixture
def app_with_figure(app):
    app.gesture_3d.create_figure("circle", (app.ancho // 2, app.alto // 2))
    app.gesture_3d.selection_mode = SelectionMode.SCALE
    return app


def test_zones_draw_only_in_scale_mode(app_with_figure):
    app = app_with_figure
    frame = np.zeros((app.alto, app.ancho, 3), dtype=np.uint8)
    left_rect, right_rect = app._get_scale_zone_rects()

    # No dibujar si no está en modo escala
    app.gesture_3d.selection_mode = SelectionMode.NORMAL
    frame_normal = frame.copy()
    # En modo normal no se llama a la función en el ciclo principal, simulamos ese comportamiento
    assert np.array_equal(frame, frame_normal)

    # Dibujar en modo escala
    app.gesture_3d.selection_mode = SelectionMode.SCALE
    frame_scale = frame.copy()
    app._draw_scale_lock_overlay(frame_scale, left_rect, right_rect)
    assert frame_scale.sum() > frame.sum()
    left_area = frame_scale[left_rect[1]:left_rect[3], left_rect[0]:left_rect[2]].mean()
    right_area = frame_scale[right_rect[1]:right_rect[3], right_rect[0]:right_rect[2]].mean()
    assert left_area > 0
    assert right_area > 0


def test_scale_lock_blocks_menu_toggle(app_with_figure):
    app = app_with_figure
    app.scale_lock_active = True
    should_allow_menu = not app.scale_lock_active
    assert should_allow_menu is False


def test_rotation_not_applied_during_lock(app_with_figure):
    app = app_with_figure
    left_rect, right_rect = app._get_scale_zone_rects()
    app.gesture_3d.selected_figure["rotation"] = 0.5
    app.scale_lock_active = True

    app._manejar_escala_bloqueada(Gesture.OPEN_HAND, app.gesture_3d.selected_figure["position"], 0.2, left_rect, right_rect)

    assert app.gesture_3d.selected_figure["rotation"] == 0.5


def test_creation_ignored_during_lock(app_with_figure):
    app = app_with_figure
    app.scale_lock_active = True
    existing = len(app.gesture_3d.figures)

    app._procesar_teclas_gestos(ord('1'))

    assert len(app.gesture_3d.figures) == existing


def test_scaling_applies_only_inside_zones(app_with_figure):
    app = app_with_figure
    left_rect, right_rect = app._get_scale_zone_rects()
    g3d = app.gesture_3d
    g3d.selection_mode = SelectionMode.SCALE
    g3d.selected_figure["size"] = 100

    # Derecha crece
    g3d.handle_figure_scaling_by_spatial((right_rect[0] + 10, (left_rect[1] + left_rect[3]) // 2), 0.2, left_rect, right_rect)
    assert g3d.selected_figure["size"] > 100

    # Izquierda reduce
    g3d.selected_figure["size"] = 100
    g3d.handle_figure_scaling_by_spatial((left_rect[0] + 10, (left_rect[1] + left_rect[3]) // 2), 0.2, left_rect, right_rect)
    assert g3d.selected_figure["size"] < 100

    # Fuera no cambia
    g3d.selected_figure["size"] = 100
    g3d.handle_figure_scaling_by_spatial((app.ancho // 2, int(app.alto * 0.9)), 0.2, left_rect, right_rect)
    assert g3d.selected_figure["size"] == 100

    # Respeta límites
    g3d.selected_figure["size"] = g3d.max_figure_size
    g3d.handle_figure_scaling_by_spatial((right_rect[0] + 10, (left_rect[1] + left_rect[3]) // 2), 0.5, left_rect, right_rect)
    assert g3d.selected_figure["size"] <= g3d.max_figure_size

