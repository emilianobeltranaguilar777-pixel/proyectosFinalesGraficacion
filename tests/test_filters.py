"""Tests para filters.py - Filtro EMA."""
import math
import pytest
import numpy as np

from filters import EMAFilter


class TestEMAFilter:
    """Tests para la clase EMAFilter."""

    def test_ema_converges_on_noisy_signal(self):
        """Test: EMA converge hacia el valor real en una senal ruidosa."""
        ema = EMAFilter(alpha=0.3)

        # Senal objetivo = 100, con ruido aleatorio
        target = 100
        np.random.seed(42)
        noisy_signal = [(target + np.random.randint(-20, 20),
                         target + np.random.randint(-20, 20))
                        for _ in range(50)]

        results = []
        for point in noisy_signal:
            smoothed = ema.update(point)
            results.append(smoothed)

        # Los ultimos valores deben estar cerca del target
        last_10 = results[-10:]
        avg_x = sum(p[0] for p in last_10) / 10
        avg_y = sum(p[1] for p in last_10) / 10

        # Tolerancia de 15 pixeles (el ruido original era +/-20)
        assert abs(avg_x - target) < 15, f"X no convergio: {avg_x} vs {target}"
        assert abs(avg_y - target) < 15, f"Y no convergio: {avg_y} vs {target}"

    def test_ema_handles_none_input(self):
        """Test: EMA maneja None sin crashear y retorna ultimo valor."""
        ema = EMAFilter(alpha=0.5)

        # Caso 1: None antes de cualquier valor
        result = ema.update(None)
        assert result is None

        # Caso 2: Valor valido, luego None
        ema.update((100, 200))
        result_after_value = ema.update(None)

        # Debe retornar el ultimo valor conocido
        assert result_after_value is not None
        assert result_after_value == (100, 200)

        # Caso 3: Varios None seguidos deben mantener el valor
        for _ in range(5):
            result = ema.update(None)
        assert result == (100, 200)

    def test_ema_reduces_jitter(self):
        """Test: EMA reduce la varianza (jitter) de la senal."""
        ema = EMAFilter(alpha=0.4)

        # Senal con jitter alrededor de (100, 100)
        jittery_signal = [
            (98, 102), (103, 97), (99, 101), (102, 99),
            (97, 103), (101, 98), (100, 100), (99, 102),
            (102, 98), (98, 101)
        ]

        raw_variance_x = np.var([p[0] for p in jittery_signal])
        raw_variance_y = np.var([p[1] for p in jittery_signal])

        smoothed = [ema.update(p) for p in jittery_signal]
        smoothed_variance_x = np.var([p[0] for p in smoothed])
        smoothed_variance_y = np.var([p[1] for p in smoothed])

        # La varianza suavizada debe ser menor que la original
        assert smoothed_variance_x < raw_variance_x
        assert smoothed_variance_y < raw_variance_y

    def test_ema_first_value_is_passed_through(self):
        """Test: El primer valor se pasa directamente sin suavizar."""
        ema = EMAFilter(alpha=0.5)

        first = ema.update((100, 200))
        assert first == (100, 200)

    def test_ema_reset_clears_history(self):
        """Test: reset() elimina el historico."""
        ema = EMAFilter(alpha=0.5)

        ema.update((100, 100))
        ema.update((200, 200))
        assert ema.has_value is True

        ema.reset()
        assert ema.has_value is False
        assert ema.update(None) is None

    def test_ema_alpha_validation(self):
        """Test: alpha fuera de rango lanza error."""
        with pytest.raises(ValueError):
            EMAFilter(alpha=0)

        with pytest.raises(ValueError):
            EMAFilter(alpha=1.5)

        with pytest.raises(ValueError):
            EMAFilter(alpha=-0.1)

        # Estos deben funcionar
        EMAFilter(alpha=0.01)
        EMAFilter(alpha=1.0)

    def test_ema_smoothing_formula(self):
        """Test: Verificar que la formula EMA es correcta."""
        alpha = 0.5
        ema = EMAFilter(alpha=alpha)

        # Primer valor
        ema.update((100, 100))

        # Segundo valor: new = alpha * 200 + (1-alpha) * 100 = 150
        result = ema.update((200, 200))
        assert result == (150, 150)

        # Tercer valor: new = 0.5 * 200 + 0.5 * 150 = 175
        result = ema.update((200, 200))
        assert result == (175, 175)


class TestCentroidSmoothing:
    """Tests de integracion para suavizado de centroide."""

    def test_centroid_smoothing_reduces_jitter(self):
        """Test: El suavizado del centroide reduce el jitter medible."""
        from ColorPainter import ColorPainter

        painter = ColorPainter(640, 480)

        # Simular detecciones de centroide con jitter
        jittery_centroids = [
            (320, 240), (322, 238), (319, 241), (321, 239),
            (318, 242), (323, 237), (320, 240), (321, 241)
        ]

        # Aplicar filtro manualmente (como lo hace process_frame)
        smoothed = []
        for centroid in jittery_centroids:
            result = painter.centroid_filter.update(centroid)
            smoothed.append(result)

        # Calcular varianza
        raw_var = np.var([c[0] for c in jittery_centroids])
        smooth_var = np.var([c[0] for c in smoothed])

        # La varianza suavizada debe ser menor
        assert smooth_var < raw_var, \
            f"Varianza no reducida: {smooth_var} >= {raw_var}"

    def test_centroid_filter_persists_on_none(self):
        """Test: El filtro mantiene el ultimo valor cuando no hay deteccion."""
        from ColorPainter import ColorPainter

        painter = ColorPainter(640, 480)

        # Simular deteccion, luego perdida de tracking
        painter.centroid_filter.update((320, 240))
        painter.centroid_filter.update((322, 242))

        # Perdida de tracking (None)
        result = painter.centroid_filter.update(None)

        # Debe mantener una posicion valida
        assert result is not None
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)


class TestPinchSmoothing:
    """Tests de integracion para suavizado de pinch_position."""

    def test_pinch_filter_initialized(self):
        """Test: Gesture3D tiene el filtro de pinch inicializado."""
        from Gesture3D import Gesture3D

        g3d = Gesture3D(640, 480)
        g3d.mediapipe_available = False

        assert hasattr(g3d, 'pinch_filter')
        assert isinstance(g3d.pinch_filter, EMAFilter)

    def test_pinch_smoothing_reduces_jitter(self):
        """Test: El suavizado del pinch reduce el jitter."""
        from Gesture3D import Gesture3D

        g3d = Gesture3D(640, 480)
        g3d.mediapipe_available = False

        # Simular posiciones de pinch con jitter
        jittery_pinch = [
            (400, 300), (402, 298), (399, 301), (401, 299),
            (398, 302), (403, 297), (400, 300), (401, 301)
        ]

        smoothed = []
        for pos in jittery_pinch:
            result = g3d.pinch_filter.update(pos)
            smoothed.append(result)

        # Calcular varianza
        raw_var_x = np.var([p[0] for p in jittery_pinch])
        smooth_var_x = np.var([p[0] for p in smoothed])

        assert smooth_var_x < raw_var_x
