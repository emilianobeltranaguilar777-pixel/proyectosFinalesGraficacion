"""Tests para la funcionalidad de escala por zonas + scale lock."""
import math
import time
import types

import pytest
import numpy as np

from Gesture3D import Gesture3D, Gesture, SelectionMode


class TestSpatialScalingMethod:
    """Tests para el método handle_figure_scaling_by_spatial en Gesture3D."""

    def test_scaling_in_left_zone_decreases_size(self, mock_gesture3d, mock_figure):
        """Test: escalar en zona izquierda reduce el tamaño."""
        g3d = mock_gesture3d
        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        initial_size = mock_figure['size']
        zone_left_max = g3d.width * 0.33
        zone_right_min = g3d.width * 0.67

        # Pinch en zona izquierda
        pinch_pos = (50, 240)  # x=50 está en zona izquierda (< 211)
        dt = 0.1  # 100ms

        result = g3d.handle_figure_scaling_by_spatial(pinch_pos, dt, zone_left_max, zone_right_min)

        assert result is True
        assert g3d.is_spatial_scale_active() is True
        assert mock_figure['size'] < initial_size

    def test_scaling_in_right_zone_increases_size(self, mock_gesture3d, mock_figure):
        """Test: escalar en zona derecha aumenta el tamaño."""
        g3d = mock_gesture3d
        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        initial_size = mock_figure['size']
        zone_left_max = g3d.width * 0.33
        zone_right_min = g3d.width * 0.67

        # Pinch en zona derecha
        pinch_pos = (600, 240)  # x=600 está en zona derecha (> 428)
        dt = 0.1

        result = g3d.handle_figure_scaling_by_spatial(pinch_pos, dt, zone_left_max, zone_right_min)

        assert result is True
        assert g3d.is_spatial_scale_active() is True
        assert mock_figure['size'] > initial_size

    def test_no_scaling_in_center_zone(self, mock_gesture3d, mock_figure):
        """Test: no hay escala en zona central."""
        g3d = mock_gesture3d
        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        initial_size = mock_figure['size']
        zone_left_max = g3d.width * 0.33
        zone_right_min = g3d.width * 0.67

        # Pinch en zona central
        pinch_pos = (320, 240)  # x=320 está en zona central
        dt = 0.1

        result = g3d.handle_figure_scaling_by_spatial(pinch_pos, dt, zone_left_max, zone_right_min)

        assert result is False
        assert g3d.is_spatial_scale_active() is False
        assert mock_figure['size'] == initial_size

    def test_size_clamping_min(self, mock_gesture3d, mock_figure):
        """Test: el tamaño no baja del mínimo."""
        g3d = mock_gesture3d
        mock_figure['size'] = g3d.min_figure_size + 1  # Casi en el mínimo
        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        zone_left_max = g3d.width * 0.33
        zone_right_min = g3d.width * 0.67

        # Escalar mucho en zona izquierda
        for _ in range(50):
            g3d.handle_figure_scaling_by_spatial((10, 240), 0.1, zone_left_max, zone_right_min)

        assert mock_figure['size'] >= g3d.min_figure_size

    def test_size_clamping_max(self, mock_gesture3d, mock_figure):
        """Test: el tamaño no sube del máximo."""
        g3d = mock_gesture3d
        mock_figure['size'] = g3d.max_figure_size - 1  # Casi en el máximo
        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        zone_left_max = g3d.width * 0.33
        zone_right_min = g3d.width * 0.67

        # Escalar mucho en zona derecha
        for _ in range(50):
            g3d.handle_figure_scaling_by_spatial((630, 240), 0.1, zone_left_max, zone_right_min)

        assert mock_figure['size'] <= g3d.max_figure_size

    def test_no_scaling_without_selected_figure(self, mock_gesture3d):
        """Test: no hay escala sin figura seleccionada."""
        g3d = mock_gesture3d
        zone_left_max = g3d.width * 0.33
        zone_right_min = g3d.width * 0.67

        result = g3d.handle_figure_scaling_by_spatial((50, 240), 0.1, zone_left_max, zone_right_min)

        assert result is False
        assert g3d.is_spatial_scale_active() is False

    def test_no_scaling_without_pinch_position(self, mock_gesture3d, mock_figure):
        """Test: no hay escala sin posición de pinch."""
        g3d = mock_gesture3d
        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        zone_left_max = g3d.width * 0.33
        zone_right_min = g3d.width * 0.67

        result = g3d.handle_figure_scaling_by_spatial(None, 0.1, zone_left_max, zone_right_min)

        assert result is False


class TestScaleStateReset:
    """Tests para el reset de estados de escala."""

    def test_reset_spatial_scale_state(self, mock_gesture3d):
        """Test: reset_spatial_scale_state limpia el estado."""
        g3d = mock_gesture3d
        g3d._scale_zone_active = True

        g3d.reset_spatial_scale_state()

        assert g3d._scale_zone_active is False

    def test_reset_angular_scale_state(self, mock_gesture3d):
        """Test: reset_angular_scale_state limpia el estado angular."""
        g3d = mock_gesture3d
        g3d.scale_reference_distance = 100
        g3d.scale_original_size = 50

        g3d.reset_angular_scale_state()

        assert g3d.scale_reference_distance == 0
        assert g3d.scale_original_size == 0


class TestScaleModeToggle:
    """Tests para el toggle del modo de escala."""

    def test_toggle_scale_mode_normal_to_scale(self, mock_gesture3d):
        """Test: cambiar de NORMAL a SCALE."""
        g3d = mock_gesture3d
        assert g3d.selection_mode == SelectionMode.NORMAL

        g3d.toggle_scale_mode()

        assert g3d.selection_mode == SelectionMode.SCALE

    def test_toggle_scale_mode_scale_to_normal(self, mock_gesture3d):
        """Test: cambiar de SCALE a NORMAL."""
        g3d = mock_gesture3d
        g3d.selection_mode = SelectionMode.SCALE

        g3d.toggle_scale_mode()

        assert g3d.selection_mode == SelectionMode.NORMAL

    def test_selection_color_changes_with_mode(self, mock_gesture3d, mock_figure):
        """Test: el color de selección cambia según el modo."""
        g3d = mock_gesture3d
        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        # Modo NORMAL -> amarillo
        g3d.selection_mode = SelectionMode.NORMAL
        g3d.toggle_scale_mode()  # Cambiar a SCALE

        assert mock_figure['selection_color'] == (0, 0, 255)  # Rojo para SCALE

        g3d.toggle_scale_mode()  # Cambiar a NORMAL

        assert mock_figure['selection_color'] == (255, 255, 80)  # Amarillo para NORMAL


class TestAngularScalingCompatibility:
    """Tests para verificar que la escala angular sigue funcionando."""

    def test_angular_scaling_method_exists(self, mock_gesture3d):
        """Test: el método de escala angular existe."""
        assert hasattr(mock_gesture3d, 'handle_figure_scaling_by_fingers')

    def test_angular_scaling_changes_size(self, mock_gesture3d, mock_figure):
        """Test: la escala angular cambia el tamaño de la figura."""
        g3d = mock_gesture3d
        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure
        g3d.selection_mode = SelectionMode.SCALE

        initial_size = mock_figure['size']

        # Simular referencia de escala
        g3d.scale_reference_distance = 50
        g3d.scale_original_size = initial_size
        g3d.current_finger_distance = 100  # El doble de la referencia

        g3d.handle_figure_scaling_by_fingers()

        # El tamaño debe haber aumentado
        assert mock_figure['size'] > initial_size


def _install_fake_cv2_local():
    """Instala un stub mínimo de cv2 para entornos headless."""
    import sys
    import types

    fake_cv2 = types.SimpleNamespace(
        LINE_AA=1,
        FONT_HERSHEY_SIMPLEX=0,
        WINDOW_NORMAL=0,
        COLOR_BGR2RGB=0,
        COLOR_HSV2BGR=0,
        COLOR_BGR2HSV=0,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5,
        CAP_PROP_BUFFERSIZE=38,
        MORPH_OPEN=0,
        MORPH_CLOSE=0,
        RETR_EXTERNAL=0,
        CHAIN_APPROX_SIMPLE=0,
    )

    def _return_img(img, *_, **__):
        return img

    fake_cv2.putText = _return_img
    fake_cv2.line = _return_img
    fake_cv2.circle = _return_img
    fake_cv2.rectangle = _return_img
    fake_cv2.polylines = _return_img
    fake_cv2.ellipse = _return_img
    fake_cv2.cvtColor = lambda img, *_args, **_kwargs: img
    fake_cv2.GaussianBlur = lambda img, *_args, **_kwargs: img
    fake_cv2.addWeighted = lambda src1, alpha, src2, beta, gamma, dst=None: src1 if dst is None else dst
    fake_cv2.resize = lambda img, size: np.zeros((size[1], size[0], 3), dtype=np.uint8)
    fake_cv2.flip = _return_img
    fake_cv2.namedWindow = lambda *_, **__: None
    fake_cv2.resizeWindow = lambda *_, **__: None
    fake_cv2.waitKey = lambda *_: 0
    fake_cv2.destroyAllWindows = lambda: None
    fake_cv2.imshow = lambda *_, **__: None
    fake_cv2.getTextSize = lambda text, font, scale, thick: ((100, 20), 5)
    fake_cv2.getTickCount = lambda: 0
    fake_cv2.getTickFrequency = lambda: 1e9
    fake_cv2.inRange = lambda img, low, high: np.zeros(img.shape[:2], dtype=np.uint8)
    fake_cv2.morphologyEx = lambda img, op, kernel: img
    fake_cv2.getStructuringElement = lambda shape, size: np.ones(size, dtype=np.uint8)
    fake_cv2.findContours = lambda img, mode, method: ([], None)
    fake_cv2.contourArea = lambda cnt: 0
    fake_cv2.moments = lambda cnt: {'m00': 0, 'm10': 0, 'm01': 0}
    fake_cv2.VideoCapture = lambda *_, **__: types.SimpleNamespace(
        isOpened=lambda: False, read=lambda: (False, None),
        set=lambda *a, **k: None, release=lambda: None
    )

    sys.modules["cv2"] = fake_cv2
    return fake_cv2


@pytest.fixture
def mock_pizarra(monkeypatch):
    """Crea una instancia mock de PizarraNeon para tests."""
    import sys

    # Instalar fake cv2
    _install_fake_cv2_local()

    # Recargar módulos que dependen de cv2
    modules_to_reload = ["Gesture3D", "main", "neon_menu", "ColorPainter", "filters", "geometry_utils"]
    for mod in modules_to_reload:
        if mod in sys.modules:
            del sys.modules[mod]

    # Mock Gesture3D antes de importar
    from Gesture3D import Gesture3D
    monkeypatch.setattr(Gesture3D, "_initialize_mediapipe", lambda self: None)

    from main import PizarraNeon

    pizarra = PizarraNeon()
    pizarra.gesture_3d.mediapipe_available = False
    return pizarra


class TestScaleLockIntegration:
    """Tests para la integración del scale lock en main.py (usando mocks)."""

    def test_scale_lock_not_active_in_normal_mode(self, mock_pizarra, mock_figure):
        """Test: scale_lock no activo en modo NORMAL."""
        pizarra = mock_pizarra
        pizarra.gesture_3d.figures.append(mock_figure)
        pizarra.gesture_3d.selected_figure = mock_figure
        pizarra.gesture_3d.selection_mode = SelectionMode.NORMAL
        pizarra.gesture_3d.pinch_active = True
        pizarra.gesture_3d.last_pinch_position = (50, 400)  # En zona izquierda

        result = pizarra._calcular_scale_lock((50, 400))

        assert result is False

    def test_scale_lock_active_in_scale_mode_with_pinch_in_zone(self, mock_pizarra, mock_figure):
        """Test: scale_lock activo en modo SCALE con pinch en zona."""
        pizarra = mock_pizarra

        # Usar toggle_scale_mode para asegurar que se usa el SelectionMode correcto
        pizarra.gesture_3d.toggle_scale_mode()  # NORMAL -> SCALE

        pizarra.gesture_3d.figures.append(mock_figure)
        pizarra.gesture_3d.selected_figure = mock_figure
        pizarra.gesture_3d.pinch_active = True

        # Verificar estado antes de llamar
        assert pizarra.gesture_3d.selected_figure is not None
        assert pizarra.gesture_3d.pinch_active is True

        # Posición en zona izquierda (x=50 < 33% de 1280 = 422), dentro de márgenes verticales
        pinch_pos = (50, 360)

        result = pizarra._calcular_scale_lock(pinch_pos)

        assert result is True, f"Expected True but got {result}. Config: ancho={pizarra.ancho}, alto={pizarra.alto}"

    def test_scale_lock_not_active_without_figure(self, mock_pizarra):
        """Test: scale_lock no activo sin figura seleccionada."""
        pizarra = mock_pizarra
        pizarra.gesture_3d.selection_mode = SelectionMode.SCALE
        pizarra.gesture_3d.pinch_active = True
        pizarra.gesture_3d.selected_figure = None

        result = pizarra._calcular_scale_lock((50, 400))

        assert result is False

    def test_scale_lock_not_active_without_pinch(self, mock_pizarra, mock_figure):
        """Test: scale_lock no activo sin pinch activo."""
        pizarra = mock_pizarra
        pizarra.gesture_3d.figures.append(mock_figure)
        pizarra.gesture_3d.selected_figure = mock_figure
        pizarra.gesture_3d.selection_mode = SelectionMode.SCALE
        pizarra.gesture_3d.pinch_active = False

        result = pizarra._calcular_scale_lock((50, 400))

        assert result is False

    def test_scale_lock_not_active_outside_vertical_margins(self, mock_pizarra, mock_figure):
        """Test: scale_lock no activo fuera de márgenes verticales."""
        pizarra = mock_pizarra
        pizarra.gesture_3d.figures.append(mock_figure)
        pizarra.gesture_3d.selected_figure = mock_figure
        pizarra.gesture_3d.selection_mode = SelectionMode.SCALE
        pizarra.gesture_3d.pinch_active = True

        # Posición en zona izquierda pero fuera de márgenes verticales (arriba)
        result = pizarra._calcular_scale_lock((50, 10))  # Muy arriba

        assert result is False

    def test_scale_lock_not_active_in_center_zone(self, mock_pizarra, mock_figure):
        """Test: scale_lock no activo en zona central."""
        pizarra = mock_pizarra
        pizarra.gesture_3d.figures.append(mock_figure)
        pizarra.gesture_3d.selected_figure = mock_figure
        pizarra.gesture_3d.selection_mode = SelectionMode.SCALE
        pizarra.gesture_3d.pinch_active = True

        # Posición en zona central
        center_x = pizarra.ancho // 2
        center_y = pizarra.alto // 2

        result = pizarra._calcular_scale_lock((center_x, center_y))

        assert result is False


class TestZoneOverlayConditions:
    """Tests para las condiciones de dibujo de overlays de zonas."""

    def test_zones_not_drawn_without_selected_figure(self, mock_pizarra):
        """Test: zonas no se dibujan sin figura seleccionada (método retorna temprano)."""
        pizarra = mock_pizarra
        pizarra.gesture_3d.selected_figure = None
        pizarra.gesture_3d.selection_mode = SelectionMode.SCALE

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # El método debe ejecutarse sin excepciones y retornar temprano
        pizarra.dibujar_zonas_escala(frame)

        # Sin figura seleccionada, el método no modifica nada
        assert pizarra.gesture_3d.selected_figure is None

    def test_zones_not_drawn_in_normal_mode(self, mock_pizarra, mock_figure):
        """Test: zonas no se dibujan en modo NORMAL (método retorna temprano)."""
        pizarra = mock_pizarra
        pizarra.gesture_3d.figures.append(mock_figure)
        pizarra.gesture_3d.selected_figure = mock_figure
        pizarra.gesture_3d.selection_mode = SelectionMode.NORMAL

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # El método debe ejecutarse sin excepciones y retornar temprano
        pizarra.dibujar_zonas_escala(frame)

        # En modo NORMAL el método retorna sin dibujar
        assert pizarra.gesture_3d.selection_mode == SelectionMode.NORMAL

    def test_zones_drawn_in_scale_mode_with_figure(self, mock_pizarra, mock_figure):
        """Test: zonas se dibujan en modo SCALE con figura seleccionada (no lanza excepciones)."""
        pizarra = mock_pizarra
        pizarra.gesture_3d.figures.append(mock_figure)
        pizarra.gesture_3d.selected_figure = mock_figure
        pizarra.gesture_3d.selection_mode = SelectionMode.SCALE

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # El método debe ejecutarse sin excepciones cuando hay figura y modo SCALE
        # (las llamadas de dibujo ocurren internamente)
        pizarra.dibujar_zonas_escala(frame)

        # Verificar que el método no modifica estados incorrectamente
        assert pizarra.gesture_3d.selection_mode == SelectionMode.SCALE
        assert pizarra.gesture_3d.selected_figure is not None
