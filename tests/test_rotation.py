"""Tests para la logica de rotacion en Gesture3D."""
import math
import time

import pytest

from Gesture3D import Gesture3D, Gesture, SelectionMode


class TestRotationFreeze:
    """Tests para verificar que la rotacion se congela correctamente."""

    def test_rotation_freeze_when_open_hand_stops(self, mock_gesture3d, mock_figure):
        """Test: la rotacion se congela cuando OPEN_HAND deja de estar activo."""
        g3d = mock_gesture3d

        # Crear y seleccionar figura
        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        initial_rotation = mock_figure['rotation']

        # Simular OPEN_HAND activo por un tiempo
        g3d.handle_gestures(Gesture.OPEN_HAND, None, time.time())
        time.sleep(0.05)  # Pequeña pausa para dt
        g3d.handle_gestures(Gesture.OPEN_HAND, None, time.time())

        rotation_after_open = mock_figure['rotation']

        # Debe haber rotado
        assert rotation_after_open > initial_rotation

        # Simular que se cierra la mano (FIST o NONE)
        g3d.handle_gestures(Gesture.FIST, None, time.time())
        rotation_after_fist = mock_figure['rotation']

        # Simular varios frames sin OPEN_HAND
        for _ in range(5):
            time.sleep(0.02)
            g3d.handle_gestures(Gesture.NONE, None, time.time())

        rotation_final = mock_figure['rotation']

        # La rotacion debe estar congelada (igual que cuando se cerro)
        assert rotation_final == rotation_after_fist

    def test_rotation_continuous_while_open_hand(self, mock_gesture3d, mock_figure):
        """Test: la rotacion es continua mientras OPEN_HAND esta activo."""
        g3d = mock_gesture3d

        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        rotations = []

        # Simular OPEN_HAND por varios frames
        for _ in range(5):
            g3d.handle_gestures(Gesture.OPEN_HAND, None, time.time())
            rotations.append(mock_figure['rotation'])
            time.sleep(0.03)  # ~33ms entre frames

        # Cada rotacion debe ser mayor que la anterior
        for i in range(1, len(rotations)):
            assert rotations[i] > rotations[i-1], \
                f"Rotacion no es continua: {rotations[i]} <= {rotations[i-1]}"


class TestNoPinchRotationMutex:
    """Tests para verificar mutex entre PINCH y OPEN_HAND."""

    def test_no_rotation_during_pinch(self, mock_gesture3d, mock_figure):
        """Test: no debe rotar mientras PINCH esta activo."""
        g3d = mock_gesture3d

        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        initial_rotation = mock_figure['rotation']

        # Activar PINCH primero
        pinch_pos = (100, 100)
        g3d.handle_gestures(Gesture.PINCH, pinch_pos, time.time())

        assert g3d.pinch_active is True

        # Guardar rotacion despues de activar pinch
        rotation_after_pinch_start = mock_figure['rotation']

        # Simular frames donde podria haber OPEN_HAND pero PINCH sigue activo
        # (En la implementacion real, si PINCH esta activo, OPEN_HAND no se detecta)
        # Pero verificamos que la logica del mutex funciona
        for _ in range(5):
            # Simular que seguimos con PINCH activo
            g3d.handle_gestures(Gesture.PINCH, pinch_pos, time.time())
            time.sleep(0.02)

        rotation_final = mock_figure['rotation']

        # La rotacion no debe haber cambiado durante PINCH
        assert rotation_final == rotation_after_pinch_start == initial_rotation

    def test_rotation_resumes_after_pinch_release(self, mock_gesture3d, mock_figure):
        """Test: la rotacion puede continuar despues de soltar PINCH."""
        g3d = mock_gesture3d

        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        # Activar y luego soltar PINCH
        g3d.handle_gestures(Gesture.PINCH, (100, 100), time.time())
        assert g3d.pinch_active is True

        g3d.handle_gestures(Gesture.NONE, None, time.time())
        assert g3d.pinch_active is False

        rotation_before = mock_figure['rotation']

        # Ahora OPEN_HAND debe poder rotar
        time.sleep(0.02)
        g3d.handle_gestures(Gesture.OPEN_HAND, None, time.time())

        rotation_after = mock_figure['rotation']

        # Debe haber rotado
        assert rotation_after > rotation_before


class TestRotationSpeed:
    """Tests para la velocidad de rotacion."""

    def test_rotation_speed_default(self, mock_gesture3d):
        """Test: velocidad de rotacion por defecto es pi rad/s."""
        assert mock_gesture3d.rotation_speed == math.pi

    def test_rotation_amount_proportional_to_dt(self, mock_gesture3d, mock_figure):
        """Test: la cantidad de rotacion es proporcional a deltaTime."""
        g3d = mock_gesture3d

        g3d.figures.append(mock_figure)
        g3d.selected_figure = mock_figure

        # Simular un dt conocido
        dt = 0.1  # 100ms
        expected_rotation = g3d.rotation_speed * dt

        g3d.rotate_figure_continuous(dt)

        # Verificar que la rotacion es aproximadamente la esperada
        assert abs(mock_figure['rotation'] - expected_rotation) < 0.01


class TestNoRotationWithoutSelection:
    """Tests para verificar que no se rota sin figura seleccionada."""

    def test_no_rotation_without_selected_figure(self, mock_gesture3d):
        """Test: no debe rotar si no hay figura seleccionada."""
        g3d = mock_gesture3d

        assert g3d.selected_figure is None

        # Intentar rotar con OPEN_HAND
        g3d.handle_gestures(Gesture.OPEN_HAND, None, time.time())

        # No debe haber error y no debe haber cambios
        assert g3d.selected_figure is None
