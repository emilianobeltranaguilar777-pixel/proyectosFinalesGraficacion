"""Neon circular menu with glow and gesture-friendly interactions."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - handled in tests via headless mode
    import cv2  # type: ignore
except Exception:  # ImportError or missing system deps
    cv2 = None

import numpy as np


Color = Tuple[int, int, int]


class MenuState(Enum):
    """Lifecycle states for the neon menu."""

    HIDDEN = auto()
    OPENING = auto()
    VISIBLE = auto()
    CLOSING = auto()


@dataclass
class MenuButton:
    """Configuration for a single radial button."""

    label: str
    color: Color
    on_select: Optional[Callable[["MenuButton"], None]] = None
    hover_level: float = field(default=0.0, init=False)
    flash_level: float = field(default=0.0, init=False)


class NeonMenu:
    """Animated, neon-styled radial menu rendered with OpenCV."""

    def __init__(
        self,
        center: Tuple[int, int] = (320, 240),
        radius: float = 85.0,
        buttons: Optional[Sequence[MenuButton]] = None,
        inner_deadzone: float = 26.0,
        start_angle: float = -math.pi / 2,
        button_radius: float = 20.0,
        open_duration: float = 0.35,
        close_duration: float = 0.25,
        glow_intensity: float = 0.7,
        hover_fade: float = 0.25,
        flash_duration: float = 0.35,
        pulse_speed: float = 2.2,
    ) -> None:
        self.center = center
        self.radius = radius
        self.inner_deadzone = inner_deadzone
        self.start_angle = start_angle
        self.button_radius = button_radius
        self.open_duration = max(open_duration, 0.01)
        self.close_duration = max(close_duration, 0.01)
        self.glow_intensity = glow_intensity
        self.hover_fade = max(hover_fade, 0.01)
        self.flash_duration = max(flash_duration, 0.01)
        self.pulse_speed = pulse_speed

        self.buttons: List[MenuButton] = list(buttons or [])
        self.state: MenuState = MenuState.HIDDEN
        self.animation: float = 0.0
        self._hover_index: Optional[int] = None
        self._prev_selecting: bool = False
        self._time_accum: float = 0.0
        self._cached_positions: List[Tuple[int, int]] = []
        self._last_position_state: Tuple[Tuple[int, int], float, float, int, float] = (
            self.center,
            self.radius,
            0.0,
            len(self.buttons),
            self.start_angle,
        )

    def open(self) -> None:
        """Begin opening animation."""

        if self.state in (MenuState.OPENING, MenuState.VISIBLE):
            return
        self.state = MenuState.OPENING

    def close(self) -> None:
        """Begin closing animation."""

        if self.state in (MenuState.CLOSING, MenuState.HIDDEN):
            return
        self.state = MenuState.CLOSING

    def is_visible(self) -> bool:
        """Return True if menu is fully or partially visible."""

        return self.state in (MenuState.OPENING, MenuState.VISIBLE, MenuState.CLOSING) and self.animation > 0.0

    def hit_test(self, cursor_pos: Tuple[float, float]) -> Optional[int]:
        """Public wrapper for radial hit testing."""

        return self._hit_test(cursor_pos)

    def update(self, cursor_pos: Tuple[float, float], is_selecting: bool, dt: float) -> None:
        """Update animation, hover and selection state.

        Parameters
        ----------
        cursor_pos: tuple
            Current cursor position (x, y) in pixels.
        is_selecting: bool
            True on selection gesture/button press. Selection triggers on press edge.
        dt: float
            Delta time in seconds since last frame.
        """

        self._time_accum += dt
        self._update_animation(dt)

        if not self.is_visible():
            self._hover_index = None
            self._prev_selecting = is_selecting
            return

        hover_index = self._hit_test(cursor_pos)
        self._hover_index = hover_index

        for idx, button in enumerate(self.buttons):
            target = 1.0 if idx == hover_index else 0.0
            button.hover_level = self._approach(button.hover_level, target, dt / self.hover_fade)
            if button.flash_level > 0.0:
                button.flash_level = max(0.0, button.flash_level - dt / self.flash_duration)

        if is_selecting and not self._prev_selecting and hover_index is not None:
            selected = self.buttons[hover_index]
            selected.flash_level = 1.0
            if selected.on_select:
                selected.on_select(selected)
        self._prev_selecting = is_selecting

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Render the neon menu onto the provided frame."""

        if cv2 is None:
            raise ImportError("OpenCV is required for rendering the neon menu")

        if self.animation <= 0.0:
            return frame

        num_buttons = len(self.buttons)
        if num_buttons == 0:
            return frame

        scale = self._ease_out_back(self.animation)
        sector_angle = (2 * math.pi) / num_buttons
        positions = self._get_cached_positions(scale, num_buttons, sector_angle)

        # Fondo del menú (círculo base y contorno delgado)
        base_radius = max(12, int(self.radius * 0.45 * scale))
        cv2.circle(frame, self.center, base_radius, (18, 18, 30), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, self.center, base_radius, (70, 120, 200), thickness=1, lineType=cv2.LINE_AA)

        for idx, (button, pos) in enumerate(zip(self.buttons, positions)):
            pulse = 0.05 * math.sin(self._time_accum * self.pulse_speed + idx)
            hover_scale = 1.0 + 0.14 * button.hover_level + pulse * button.hover_level
            flash_scale = 1.0 + 0.18 * button.flash_level
            size = max(6, int(self.button_radius * scale * hover_scale * flash_scale))

            base_r, base_g, base_b = button.color
            intensity_boost = 1.0 + 0.25 * button.hover_level + 0.6 * button.flash_level
            color = (
                min(255, int(base_r * intensity_boost + 40 * button.flash_level)),
                min(255, int(base_g * intensity_boost + 40 * button.flash_level)),
                min(255, int(base_b * intensity_boost + 40 * button.flash_level)),
            )

            outer_color = (
                min(255, int(color[0] * 0.8 + 30)),
                min(255, int(color[1] * 0.8 + 30)),
                min(255, int(color[2] * 0.8 + 30)),
            )

            cv2.circle(frame, pos, size, color, thickness=-1, lineType=cv2.LINE_AA)
            if button.hover_level > 0.05 or button.flash_level > 0.0:
                cv2.circle(frame, pos, int(size * 1.3), outer_color, thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(frame, pos, int(size * 1.55), outer_color, thickness=1, lineType=cv2.LINE_AA)

            self._draw_icon(frame, pos, size, button)

        return frame

    def _update_animation(self, dt: float) -> None:
        if self.state == MenuState.OPENING:
            self.animation = min(1.0, self.animation + dt / self.open_duration)
            if self.animation >= 1.0:
                self.state = MenuState.VISIBLE
        elif self.state == MenuState.CLOSING:
            self.animation = max(0.0, self.animation - dt / self.close_duration)
            if self.animation <= 0.0:
                self.state = MenuState.HIDDEN
        elif self.state == MenuState.HIDDEN:
            self.animation = 0.0

    def _hit_test(self, cursor_pos: Tuple[float, float]) -> Optional[int]:
        if self.animation <= 0.0 or not self.buttons:
            return None
        dx = cursor_pos[0] - self.center[0]
        dy = cursor_pos[1] - self.center[1]
        distance = math.hypot(dx, dy)
        effective_radius = self.radius * self.animation
        if distance < self.inner_deadzone or distance > effective_radius * 1.2:
            return None

        angle = math.atan2(dy, dx)
        angle = (angle - self.start_angle) % (2 * math.pi)
        sector_angle = (2 * math.pi) / len(self.buttons)
        index = int(angle // sector_angle)
        return max(0, min(len(self.buttons) - 1, index))

    def _polar_to_cartesian(self, angle: float, radius: float) -> Tuple[int, int]:
        x = int(self.center[0] + math.cos(angle) * radius)
        y = int(self.center[1] + math.sin(angle) * radius)
        return (x, y)

    def _get_cached_positions(
        self, scale: float, num_buttons: int, sector_angle: float
    ) -> List[Tuple[int, int]]:
        key = (self.center, self.radius, round(scale, 3), num_buttons, self.start_angle)
        if key != self._last_position_state:
            self._cached_positions = []
            for idx in range(num_buttons):
                angle = self.start_angle + sector_angle * (idx + 0.5)
                self._cached_positions.append(self._polar_to_cartesian(angle, self.radius * scale))
            self._last_position_state = key
        return self._cached_positions

    @staticmethod
    def _ease_out_back(t: float) -> float:
        c1 = 1.70158
        c3 = c1 + 1
        t = max(0.0, min(1.0, t))
        return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)

    @staticmethod
    def _approach(current: float, target: float, step: float) -> float:
        if current < target:
            return min(target, current + step)
        return max(target, current - step)

    def _draw_icon(self, overlay: np.ndarray, pos: Tuple[int, int], size: int, button: MenuButton) -> None:
        icon_color = (220, 220, 220)
        thickness = max(1, size // 6)
        r = max(4, size // 2)
        if button.label.lower() == "circle":
            cv2.circle(overlay, pos, r, icon_color, thickness, lineType=cv2.LINE_AA)
        elif button.label.lower() == "square":
            half = int(r * 0.9)
            pts = np.array(
                [
                    (pos[0] - half, pos[1] - half),
                    (pos[0] + half, pos[1] - half),
                    (pos[0] + half, pos[1] + half),
                    (pos[0] - half, pos[1] + half),
                ],
                dtype=np.int32,
            )
            cv2.polylines(overlay, [pts], True, icon_color, thickness, lineType=cv2.LINE_AA)
        elif button.label.lower() == "triangle":
            pts = np.array(
                [
                    (pos[0], pos[1] - r),
                    (pos[0] + int(r * 0.9), pos[1] + int(r * 0.9)),
                    (pos[0] - int(r * 0.9), pos[1] + int(r * 0.9)),
                ],
                dtype=np.int32,
            )
            cv2.polylines(overlay, [pts], True, icon_color, thickness, lineType=cv2.LINE_AA)
        else:
            inner = int(r * 0.5)
            cv2.circle(overlay, pos, inner, icon_color, thickness, lineType=cv2.LINE_AA)
