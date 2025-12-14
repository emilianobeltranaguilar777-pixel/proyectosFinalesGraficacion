"""Filtros de suavizado temporal para reducir jitter."""
from typing import Tuple, Optional


class EMAFilter:
    """
    Exponential Moving Average filter para suavizar coordenadas 2D.

    El filtro aplica la formula:
        smoothed = alpha * new_value + (1 - alpha) * previous_value

    Un alpha mas alto (cercano a 1) da menos suavizado pero menor latencia.
    Un alpha mas bajo (cercano a 0) da mas suavizado pero mayor latencia.
    """

    def __init__(self, alpha: float = 0.4):
        """
        Inicializa el filtro EMA.

        Args:
            alpha: Factor de suavizado entre 0 y 1.
                   Default 0.4 balancea jitter vs latencia.
        """
        if not 0 < alpha <= 1:
            raise ValueError("alpha debe estar en el rango (0, 1]")

        self.alpha = alpha
        self._value: Optional[Tuple[float, float]] = None

    def update(self, value: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Actualiza el filtro con un nuevo valor.

        Args:
            value: Nueva coordenada (x, y) o None si no hay deteccion.

        Returns:
            Coordenada suavizada (x, y) o None si no hay historico.
        """
        if value is None:
            # Si no hay valor nuevo, retornamos el ultimo conocido
            # (permite mantener posicion cuando se pierde tracking momentaneamente)
            if self._value is not None:
                return (int(self._value[0]), int(self._value[1]))
            return None

        if self._value is None:
            # Primera medicion, inicializar directamente
            self._value = (float(value[0]), float(value[1]))
        else:
            # Aplicar EMA: new = alpha * measured + (1-alpha) * previous
            self._value = (
                self.alpha * value[0] + (1 - self.alpha) * self._value[0],
                self.alpha * value[1] + (1 - self.alpha) * self._value[1]
            )

        return (int(self._value[0]), int(self._value[1]))

    def reset(self):
        """Resetea el filtro, eliminando el historico."""
        self._value = None

    @property
    def has_value(self) -> bool:
        """Retorna True si el filtro tiene un valor almacenado."""
        return self._value is not None

    @property
    def raw_value(self) -> Optional[Tuple[float, float]]:
        """Retorna el valor interno sin redondear (para debug)."""
        return self._value
