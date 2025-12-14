"""Tests para color tracking en ColorPainter."""
import math
import numpy as np
import cv2
import pytest

from ColorPainter import ColorPainter, HSV_PRESETS


class TestColorDetection:
    """Tests para deteccion de color por HSV."""

    def _create_colored_circle_image(self, width, height, center, radius, bgr_color):
        """Crea una imagen con un circulo de color especifico."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(img, center, radius, bgr_color, -1)
        return img

    def _hsv_to_bgr(self, h, s, v):
        """Convierte HSV a BGR."""
        hsv_pixel = np.uint8([[[h, s, v]]])
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in bgr_pixel[0, 0])

    def test_detect_blue_circle_centroid(self):
        """Test: detectar centroide de un circulo azul."""
        painter = ColorPainter(640, 480)

        # Crear imagen con circulo azul en (320, 240)
        expected_center = (320, 240)
        blue_bgr = self._hsv_to_bgr(115, 200, 200)  # Azul en HSV
        frame = self._create_colored_circle_image(640, 480, expected_center, 50, blue_bgr)

        # Detectar
        centroid, contour = painter.detect_color_object(frame)

        # Verificar centroide cerca del centro esperado
        assert centroid is not None
        assert abs(centroid[0] - expected_center[0]) <= 5
        assert abs(centroid[1] - expected_center[1]) <= 5

    def test_detect_green_circle_with_preset(self):
        """Test: detectar circulo verde con preset verde."""
        painter = ColorPainter(640, 480)
        painter.set_hsv_preset(2)  # Verde

        # Crear imagen con circulo verde
        expected_center = (400, 300)
        green_bgr = self._hsv_to_bgr(60, 200, 200)  # Verde en HSV
        frame = self._create_colored_circle_image(640, 480, expected_center, 40, green_bgr)

        centroid, contour = painter.detect_color_object(frame)

        assert centroid is not None
        assert abs(centroid[0] - expected_center[0]) <= 5
        assert abs(centroid[1] - expected_center[1]) <= 5

    def test_detect_yellow_circle_with_preset(self):
        """Test: detectar circulo amarillo con preset amarillo."""
        painter = ColorPainter(640, 480)
        painter.set_hsv_preset(4)  # Amarillo

        expected_center = (200, 350)
        yellow_bgr = self._hsv_to_bgr(28, 200, 220)  # Amarillo en HSV
        frame = self._create_colored_circle_image(640, 480, expected_center, 35, yellow_bgr)

        centroid, contour = painter.detect_color_object(frame)

        assert centroid is not None
        assert abs(centroid[0] - expected_center[0]) <= 5
        assert abs(centroid[1] - expected_center[1]) <= 5

    def test_no_detection_wrong_color(self):
        """Test: no detectar si el color no coincide con el preset."""
        painter = ColorPainter(640, 480)
        painter.set_hsv_preset(1)  # Azul

        # Crear imagen con circulo verde (no deberia detectar)
        green_bgr = self._hsv_to_bgr(60, 200, 200)
        frame = self._create_colored_circle_image(640, 480, (320, 240), 50, green_bgr)

        centroid, contour = painter.detect_color_object(frame)

        assert centroid is None


class TestLargestContour:
    """Tests para seleccion del contorno mas grande."""

    def test_largest_contour_selected(self):
        """Test: con dos blobs, debe elegir el mas grande."""
        painter = ColorPainter(640, 480)

        # Crear imagen con dos circulos azules de diferente tamano
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Circulo pequeno en (100, 100)
        small_center = (100, 100)
        small_radius = 20

        # Circulo grande en (400, 300)
        large_center = (400, 300)
        large_radius = 60

        # Color azul en BGR
        blue_hsv = np.uint8([[[115, 200, 200]]])
        blue_bgr = cv2.cvtColor(blue_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        blue_bgr = tuple(int(c) for c in blue_bgr)

        cv2.circle(frame, small_center, small_radius, blue_bgr, -1)
        cv2.circle(frame, large_center, large_radius, blue_bgr, -1)

        # Detectar
        centroid, contour = painter.detect_color_object(frame)

        # Debe detectar el circulo grande
        assert centroid is not None
        assert abs(centroid[0] - large_center[0]) <= 5
        assert abs(centroid[1] - large_center[1]) <= 5

    def test_largest_contour_different_positions(self):
        """Test: el contorno mas grande es seleccionado independiente de posicion."""
        painter = ColorPainter(640, 480)
        painter.set_hsv_preset(2)  # Verde

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Color verde en BGR
        green_hsv = np.uint8([[[60, 200, 200]]])
        green_bgr = cv2.cvtColor(green_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        green_bgr = tuple(int(c) for c in green_bgr)

        # Circulo grande arriba-izquierda
        large_center = (100, 100)
        large_radius = 70

        # Circulo pequeno abajo-derecha
        small_center = (500, 400)
        small_radius = 25

        cv2.circle(frame, large_center, large_radius, green_bgr, -1)
        cv2.circle(frame, small_center, small_radius, green_bgr, -1)

        centroid, contour = painter.detect_color_object(frame)

        # Debe detectar el circulo grande (arriba-izquierda)
        assert centroid is not None
        assert abs(centroid[0] - large_center[0]) <= 5
        assert abs(centroid[1] - large_center[1]) <= 5


class TestHSVPresets:
    """Tests para los presets HSV."""

    def test_all_presets_defined(self):
        """Test: todos los presets 1-6 estan definidos."""
        for i in range(1, 7):
            assert i in HSV_PRESETS
            assert 'name' in HSV_PRESETS[i]
            assert 'lower' in HSV_PRESETS[i]
            assert 'upper' in HSV_PRESETS[i]

    def test_set_hsv_preset_changes_values(self):
        """Test: set_hsv_preset cambia los valores correctamente."""
        painter = ColorPainter(640, 480)

        # Cambiar a verde
        result = painter.set_hsv_preset(2)

        assert result is True
        assert painter.preset_name == 'VERDE'
        assert painter.current_preset == 2
        np.testing.assert_array_equal(
            painter.hsv_lower,
            np.array(HSV_PRESETS[2]['lower'])
        )

    def test_set_invalid_preset_fails(self):
        """Test: set_hsv_preset con preset invalido retorna False."""
        painter = ColorPainter(640, 480)

        result = painter.set_hsv_preset(99)

        assert result is False
        # Valores no deben cambiar
        assert painter.current_preset == 1

    def test_reset_to_default(self):
        """Test: reset_to_default_preset vuelve a azul."""
        painter = ColorPainter(640, 480)
        painter.set_hsv_preset(5)  # Cambiar a magenta

        painter.reset_to_default_preset()

        assert painter.preset_name == 'AZUL'
        assert painter.current_preset == 1


class TestHSVCalibration:
    """Tests para modo calibracion HSV."""

    def test_calibration_mode_toggle(self):
        """Test: toggle de modo calibracion."""
        painter = ColorPainter(640, 480)

        assert painter.hsv_calibration_mode is False

        # No podemos testear trackbars sin display, pero verificamos el estado
        # El toggle deberia cambiar el estado interno

    def test_hsv_values_after_custom_set(self):
        """Test: valores HSV se pueden setear manualmente."""
        painter = ColorPainter(640, 480)

        # Setear valores custom directamente
        painter.hsv_lower = np.array([50, 100, 100])
        painter.hsv_upper = np.array([70, 255, 255])
        painter.preset_name = 'CUSTOM'

        assert painter.preset_name == 'CUSTOM'
        np.testing.assert_array_equal(painter.hsv_lower, [50, 100, 100])
        np.testing.assert_array_equal(painter.hsv_upper, [70, 255, 255])


class TestDetectionWithMorphology:
    """Tests que verifican que morfologia y EMA funcionan."""

    def test_small_noise_filtered(self):
        """Test: ruido pequeno es filtrado por morfologia."""
        painter = ColorPainter(640, 480)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Color azul
        blue_hsv = np.uint8([[[115, 200, 200]]])
        blue_bgr = cv2.cvtColor(blue_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        blue_bgr = tuple(int(c) for c in blue_bgr)

        # Objeto principal grande
        cv2.circle(frame, (320, 240), 50, blue_bgr, -1)

        # Ruido pequeno (deberia ser filtrado por area minima o morfologia)
        cv2.circle(frame, (50, 50), 5, blue_bgr, -1)
        cv2.circle(frame, (600, 400), 3, blue_bgr, -1)

        centroid, contour = painter.detect_color_object(frame)

        # Debe detectar el objeto grande, no el ruido
        assert centroid is not None
        assert abs(centroid[0] - 320) <= 5
        assert abs(centroid[1] - 240) <= 5

    def test_ema_smoothing_applied(self):
        """Test: EMA smoothing se aplica al centroide."""
        painter = ColorPainter(640, 480)

        # El filtro EMA debe existir
        assert hasattr(painter, 'centroid_filter')
        assert painter.centroid_filter is not None

        # Simular deteccion con jitter
        blue_bgr = (200, 100, 50)

        positions = [(320, 240), (322, 238), (318, 242)]
        smoothed = []

        for pos in positions:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(frame, pos, 50, blue_bgr, -1)

            raw_pos, _ = painter.detect_color_object(frame)
            smooth_pos = painter.centroid_filter.update(raw_pos)
            if smooth_pos:
                smoothed.append(smooth_pos)

        # Verificar que tenemos valores suavizados
        assert len(smoothed) == 3
