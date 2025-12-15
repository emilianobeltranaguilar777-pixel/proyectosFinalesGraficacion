"""Tests para la escala espacial basada en zonas verticales."""

import pytest

from Gesture3D import Gesture3D, Gesture, SelectionMode


@pytest.fixture
def gesture3d(monkeypatch):
    """Crea una instancia de Gesture3D sin MediaPipe."""

    monkeypatch.setattr(Gesture3D, "_initialize_mediapipe", lambda self: None)
    g3d = Gesture3D(640, 480, use_external_menu=False)
    g3d.mediapipe_available = False
    return g3d


@pytest.fixture
def gesture3d_with_figure(gesture3d):
    """Instancia con una figura seleccionada en modo escala."""

    gesture3d.create_figure("circle", (320, 240))
    gesture3d.selection_mode = SelectionMode.SCALE
    gesture3d.selected_figure["size"] = 100
    return gesture3d


def test_scale_increases_when_hand_above_center(gesture3d_with_figure):
    """La figura crece cuando la mano está por encima del centro."""

    g3d = gesture3d_with_figure

    g3d.handle_figure_scaling_by_vertical((320, 140))

    assert g3d.selected_figure["size"] > 100


def test_scale_decreases_when_hand_below_center(gesture3d_with_figure):
    """La figura se hace pequeña cuando la mano está debajo del centro."""

    g3d = gesture3d_with_figure

    g3d.handle_figure_scaling_by_vertical((320, 400))

    assert g3d.selected_figure["size"] < 100


def test_scale_is_clamped_within_bounds(gesture3d_with_figure):
    """El factor de escala se limita al ±50% del tamaño base."""

    g3d = gesture3d_with_figure

    # Muy por encima: debería clavar en +50%
    g3d.handle_figure_scaling_by_vertical((320, -200))
    assert g3d.selected_figure["size"] == 150

    # Reiniciar referencia y escalar muy por debajo: clamped a -50%
    g3d.reset_spatial_scale_state()
    g3d.selected_figure["size"] = 100
    g3d.handle_figure_scaling_by_vertical((320, 900))
    assert g3d.selected_figure["size"] == 50


def test_rotation_disabled_during_scale_mode(gesture3d_with_figure):
    """La rotación se desactiva al entrar en modo escala y vuelve al salir."""

    g3d = gesture3d_with_figure
    g3d.rotation_enabled = True

    # Entrar en escala mantiene la rotación bloqueada
    g3d.toggle_scale_mode()  # pasa a NORMAL
    g3d.toggle_scale_mode()  # vuelve a SCALE
    assert g3d.rotation_enabled is False

    # Al salir de modo escala se restaura
    g3d.toggle_scale_mode()
    assert g3d.rotation_enabled is True
