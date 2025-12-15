"""Neon circular menu - STABLE mode: deterministic, no glow, no pulse."""
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


class NeonMenu:
    """Stable, deterministic radial menu rendered with OpenCV.

    STABLE MODE: No glow, no pulse, no sin/cos animations.
    Draws solid circles with simple borders. Positions calculated each frame.
    """

    def __init__(
        self,
        center: Tuple[int, int] = (320, 240),
        radius: float = 85.0,
        buttons: Optional[Sequence[MenuButton]] = None,
        inner_deadzone: float = 26.0,
        start_angle: float = -math.pi / 2,
        button_radius: float = 20.0,
        open_duration: float = 0.20,
        close_duration: float = 0.15,
        hover_fade: float = 0.15,
    ) -> None:
        self.center = center
        self.radius = radius
        self.inner_deadzone = inner_deadzone
        self.start_angle = start_angle
        self.button_radius = button_radius
        self.open_duration = max(open_duration, 0.01)
        self.close_duration = max(close_duration, 0.01)
        self.hover_fade = max(hover_fade, 0.01)

        self.buttons: List[MenuButton] = list(buttons or [])
        self.state: MenuState = MenuState.HIDDEN
        self.animation: float = 0.0
        self._hover_index: Optional[int] = None
        self._prev_selecting: bool = False

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
        self._update_animation(dt)

        if not self.is_visible():
            self._hover_index = None
            self._prev_selecting = is_selecting
            return

        hover_index = self._hit_test(cursor_pos)
        self._hover_index = hover_index

        # Update hover levels (simple linear approach)
        for idx, button in enumerate(self.buttons):
            target = 1.0 if idx == hover_index else 0.0
            button.hover_level = self._approach(button.hover_level, target, dt / self.hover_fade)

        # Selection on press edge
        if is_selecting and not self._prev_selecting and hover_index is not None:
            selected = self.buttons[hover_index]
            if selected.on_select:
                selected.on_select(selected)
        self._prev_selecting = is_selecting

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Render the menu onto the provided frame.

        STABLE MODE: No glow, no pulse. Simple solid circles with borders.
        Positions calculated fresh each frame (cheap for few buttons).
        """
        if cv2 is None:
            raise ImportError("OpenCV is required for rendering the neon menu")

        if self.animation <= 0.0:
            return frame

        num_buttons = len(self.buttons)
        if num_buttons == 0:
            return frame

        # Simple linear scale for animation (no ease-out-back)
        scale = min(1.0, self.animation)

        # Calculate positions fresh each frame (deterministic, no cache bugs)
        sector_angle = (2 * math.pi) / num_buttons
        positions = []
        for idx in range(num_buttons):
            angle = self.start_angle + sector_angle * (idx + 0.5)
            x = int(self.center[0] + math.cos(angle) * self.radius * scale)
            y = int(self.center[1] + math.sin(angle) * self.radius * scale)
            positions.append((x, y))

        # Draw center base circle (solid, no glow)
        base_radius = max(12, int(self.radius * 0.45 * scale))
        cv2.circle(frame, self.center, base_radius, (20, 20, 35), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, self.center, base_radius, (80, 120, 180), thickness=2, lineType=cv2.LINE_AA)

        # Draw buttons (solid circles with simple borders)
        for idx, (button, pos) in enumerate(zip(self.buttons, positions)):
            # Size based on hover only (no pulse)
            hover_scale = 1.0 + 0.15 * button.hover_level
            size = max(8, int(self.button_radius * scale * hover_scale))

            # Color with hover brightness boost
            base_r, base_g, base_b = button.color
            brightness = 1.0 + 0.3 * button.hover_level
            color = (
                min(255, int(base_r * brightness)),
                min(255, int(base_g * brightness)),
                min(255, int(base_b * brightness)),
            )

            # Solid filled circle
            cv2.circle(frame, pos, size, color, thickness=-1, lineType=cv2.LINE_AA)

            # Simple border (slightly darker)
            border_color = (
                max(0, int(color[0] * 0.6)),
                max(0, int(color[1] * 0.6)),
                max(0, int(color[2] * 0.6)),
            )
            cv2.circle(frame, pos, size, border_color, thickness=2, lineType=cv2.LINE_AA)

            # Outer ring on hover only
            if button.hover_level > 0.1:
                ring_alpha = button.hover_level
                ring_color = (
                    min(255, int(color[0] * 0.8 + 50 * ring_alpha)),
                    min(255, int(color[1] * 0.8 + 50 * ring_alpha)),
                    min(255, int(color[2] * 0.8 + 50 * ring_alpha)),
                )
                cv2.circle(frame, pos, int(size * 1.4), ring_color, thickness=1, lineType=cv2.LINE_AA)

            # Draw icon
            self._draw_icon(frame, pos, size, button)

        return frame

    def _update_animation(self, dt: float) -> None:
        """Update animation state (linear, no easing)."""
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
        """Test which button sector the cursor is in."""
        if self.animation <= 0.0 or not self.buttons:
            return None
        dx = cursor_pos[0] - self.center[0]
        dy = cursor_pos[1] - self.center[1]
        distance = math.hypot(dx, dy)
        effective_radius = self.radius * self.animation
        if distance < self.inner_deadzone or distance > effective_radius * 1.3:
            return None

        angle = math.atan2(dy, dx)
        angle = (angle - self.start_angle) % (2 * math.pi)
        sector_angle = (2 * math.pi) / len(self.buttons)
        index = int(angle // sector_angle)
        return max(0, min(len(self.buttons) - 1, index))

    @staticmethod
    def _approach(current: float, target: float, step: float) -> float:
        """Linear approach to target value."""
        if current < target:
            return min(target, current + step)
        return max(target, current - step)

    def _draw_icon(self, frame: np.ndarray, pos: Tuple[int, int], size: int, button: MenuButton) -> None:
        """Draw simple icon for button."""
        icon_color = (230, 230, 230)
        thickness = max(1, size // 5)
        r = max(4, size // 2)

        label = button.label.lower()

        if label == "circle":
            cv2.circle(frame, pos, r, icon_color, thickness, lineType=cv2.LINE_AA)
        elif label == "square":
            half = int(r * 0.85)
            cv2.rectangle(frame, (pos[0] - half, pos[1] - half),
                         (pos[0] + half, pos[1] + half), icon_color, thickness, lineType=cv2.LINE_AA)
        elif label == "triangle":
            pts = np.array([
                [pos[0], pos[1] - r],
                [pos[0] + int(r * 0.87), pos[1] + int(r * 0.5)],
                [pos[0] - int(r * 0.87), pos[1] + int(r * 0.5)],
            ], dtype=np.int32)
            cv2.polylines(frame, [pts], True, icon_color, thickness, lineType=cv2.LINE_AA)
        elif label == "star":
            # Simple 5-point star
            points = []
            for i in range(10):
                angle = -math.pi / 2 + i * math.pi / 5
                dist = r if i % 2 == 0 else r * 0.4
                points.append([int(pos[0] + dist * math.cos(angle)),
                              int(pos[1] + dist * math.sin(angle))])
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [pts], True, icon_color, thickness, lineType=cv2.LINE_AA)
        elif label == "heart":
            # Simple heart shape
            cv2.circle(frame, (pos[0] - r // 3, pos[1] - r // 4), r // 3, icon_color, thickness, lineType=cv2.LINE_AA)
            cv2.circle(frame, (pos[0] + r // 3, pos[1] - r // 4), r // 3, icon_color, thickness, lineType=cv2.LINE_AA)
            pts = np.array([
                [pos[0], pos[1] + r // 2],
                [pos[0] - r // 2 - 2, pos[1] - r // 6],
                [pos[0] + r // 2 + 2, pos[1] - r // 6],
            ], dtype=np.int32)
            cv2.polylines(frame, [pts], True, icon_color, max(1, thickness - 1), lineType=cv2.LINE_AA)
        elif label == "hexagon":
            points = []
            for i in range(6):
                angle = math.pi / 6 + i * math.pi / 3
                points.append([int(pos[0] + r * math.cos(angle)),
                              int(pos[1] + r * math.sin(angle))])
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [pts], True, icon_color, thickness, lineType=cv2.LINE_AA)
        elif label == "delete":
            # X mark for delete
            d = r // 2
            cv2.line(frame, (pos[0] - d, pos[1] - d), (pos[0] + d, pos[1] + d),
                    (100, 100, 255), thickness + 1, lineType=cv2.LINE_AA)
            cv2.line(frame, (pos[0] - d, pos[1] + d), (pos[0] + d, pos[1] - d),
                    (100, 100, 255), thickness + 1, lineType=cv2.LINE_AA)
        else:
            # Default: simple dot
            cv2.circle(frame, pos, max(3, r // 2), icon_color, -1, lineType=cv2.LINE_AA)
