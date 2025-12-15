"""Tests para la escala espacial basada en zonas horizontales."""

import pytest

from Gesture3D import Gesture3D, SelectionMode


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


def test_scale_increases_in_right_zone(gesture3d_with_figure):
    """La figura crece cuando el cursor está en la zona derecha."""

    g3d = gesture3d_with_figure
    g3d.handle_figure_scaling_by_spatial((600, 240), dt=0.5)

    assert g3d.selected_figure["size"] > 100


def test_scale_decreases_in_left_zone(gesture3d_with_figure):
    """La figura decrece cuando el cursor está en la zona izquierda."""

    g3d = gesture3d_with_figure
    g3d.handle_figure_scaling_by_spatial((40, 240), dt=0.5)

    assert g3d.selected_figure["size"] < 100


def test_scale_neutral_in_center_zone(gesture3d_with_figure):
    """Sin cambio cuando el cursor está en la zona central."""

    g3d = gesture3d_with_figure
    g3d.handle_figure_scaling_by_spatial((320, 240), dt=0.5)

    assert g3d.selected_figure["size"] == 100


def test_scale_clamped_to_bounds(gesture3d_with_figure):
    """La escala respeta min y max establecidos."""

    g3d = gesture3d_with_figure
    g3d.selected_figure["size"] = g3d.max_figure_size - 1
    g3d.handle_figure_scaling_by_spatial((640, 240), dt=1.0)
    assert g3d.selected_figure["size"] <= g3d.max_figure_size

    g3d.selected_figure["size"] = g3d.min_figure_size + 1
    g3d.handle_figure_scaling_by_spatial((0, 240), dt=1.0)
    assert g3d.selected_figure["size"] >= g3d.min_figure_size


def test_scale_does_not_affect_rotation_or_position(gesture3d_with_figure):
    """Escalar no altera rotación ni posición."""

    g3d = gesture3d_with_figure
    g3d.selected_figure["rotation"] = 1.25
    initial_position = g3d.selected_figure["position"]

    g3d.handle_figure_scaling_by_spatial((620, 240), dt=0.25)

    assert g3d.selected_figure["rotation"] == 1.25
    assert g3d.selected_figure["position"] == initial_position


def test_scale_requires_selected_figure(gesture3d):
    """Sin figura seleccionada no hace nada."""

    gesture3d.selection_mode = SelectionMode.SCALE
    gesture3d.selected_figure = None

    gesture3d.handle_figure_scaling_by_spatial((500, 240), dt=0.4)

    assert gesture3d.selected_figure is None
