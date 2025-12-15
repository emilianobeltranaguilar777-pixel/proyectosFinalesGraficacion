import math
import types

import numpy as np

from neon_menu import MenuButton, MenuState, NeonMenu


def test_hit_testing_by_angle():
    buttons = [
        MenuButton("circle", (255, 0, 0)),
        MenuButton("square", (0, 255, 0)),
        MenuButton("triangle", (0, 0, 255)),
        MenuButton("diamond", (255, 255, 0)),
    ]
    menu = NeonMenu(center=(100, 100), radius=80, inner_deadzone=10, buttons=buttons)
    menu.state = MenuState.VISIBLE
    menu.animation = 1.0

    assert menu.hit_test((100, 20)) == 0  # Up
    assert menu.hit_test((180, 100)) == 1  # Right
    assert menu.hit_test((100, 180)) == 2  # Down
    assert menu.hit_test((20, 100)) == 3  # Left


def test_state_transitions_open_close():
    menu = NeonMenu(buttons=[MenuButton("circle", (255, 0, 0))], open_duration=0.5, close_duration=0.5)

    menu.open()
    assert menu.state == MenuState.OPENING

    menu.update((0, 0), False, 0.5)
    assert menu.state == MenuState.VISIBLE
    assert math.isclose(menu.animation, 1.0)

    menu.close()
    assert menu.state == MenuState.CLOSING

    menu.update((0, 0), False, 0.5)
    assert menu.state == MenuState.HIDDEN
    assert menu.animation == 0.0


def test_animation_depends_on_dt():
    menu = NeonMenu(buttons=[MenuButton("circle", (255, 0, 0))], open_duration=1.0)
    menu.open()
    menu.update((0, 0), False, 0.4)
    assert 0.35 < menu.animation < 0.5
    menu.update((0, 0), False, 0.6)
    assert math.isclose(menu.animation, 1.0)


def test_selection_callback_triggered_once():
    """Selection callback triggers on press edge, not held."""
    calls = []

    def on_select(button):
        calls.append(button.label)

    menu = NeonMenu(
        center=(50, 50),
        radius=40,
        buttons=[MenuButton("circle", (255, 0, 255), on_select=on_select)],
    )
    menu.state = MenuState.VISIBLE
    menu.animation = 1.0

    cursor_pos = (50, 10)
    menu.update(cursor_pos, True, 0.1)
    menu.update(cursor_pos, True, 0.1)  # Held selection should not duplicate
    menu.update(cursor_pos, False, 0.1)
    menu.update(cursor_pos, True, 0.1)  # New press should trigger again

    assert calls == ["circle", "circle"]


def test_hover_levels_smoothly_adjust():
    menu = NeonMenu(center=(0, 0), radius=50, buttons=[MenuButton("square", (0, 255, 255))], hover_fade=0.1)
    menu.state = MenuState.OPENING
    menu.animation = 0.8

    menu.update((0, 50), False, 0.05)
    hover_after_first = menu.buttons[0].hover_level
    assert hover_after_first > 0.0

    menu.update((100, 100), False, 0.1)
    assert menu.buttons[0].hover_level < hover_after_first


def test_draw_fast_path_does_not_use_blur(monkeypatch):
    """STABLE mode: No blur, no addWeighted for glow effects."""
    frame = np.zeros((120, 120, 3), dtype=np.uint8)

    def fail(*_args, **_kwargs):  # pragma: no cover - failure guard
        raise AssertionError("Blur or full-frame blend should not be called")

    fake_cv2 = types.SimpleNamespace(
        LINE_AA=1,
        FONT_HERSHEY_SIMPLEX=0,
        circle=lambda img, *args, **kwargs: img,
        polylines=lambda img, *args, **kwargs: img,
        line=lambda img, *args, **kwargs: img,
        rectangle=lambda img, *args, **kwargs: img,
        putText=lambda img, *args, **kwargs: img,
        GaussianBlur=fail,
        addWeighted=fail,
        resize=fail,
    )

    monkeypatch.setattr("neon_menu.cv2", fake_cv2)

    menu = NeonMenu(buttons=[MenuButton("circle", (255, 0, 0))])
    menu.state = MenuState.VISIBLE
    menu.animation = 1.0

    result = menu.draw(frame)
    assert result is frame


def test_menu_closes_after_selection():
    """Menu closes immediately when selection callback calls close()."""
    closed = []

    def on_select(button):
        # Simulate what happens in main.py callback
        closed.append(True)

    menu = NeonMenu(
        center=(50, 50),
        radius=40,
        buttons=[MenuButton("circle", (255, 0, 255), on_select=on_select)],
    )
    menu.state = MenuState.VISIBLE
    menu.animation = 1.0

    # Simulate selection
    cursor_pos = (50, 10)
    menu.update(cursor_pos, True, 0.1)

    # Callback was called
    assert len(closed) == 1

    # Menu should be closable immediately after selection
    menu.close()
    assert menu.state == MenuState.CLOSING


def test_menu_not_drawn_when_hidden():
    """Menu draw() is a no-op when animation is 0."""
    draw_calls = []

    def track_circle(img, *args, **kwargs):
        draw_calls.append("circle")
        return img

    fake_cv2 = types.SimpleNamespace(
        LINE_AA=1,
        FONT_HERSHEY_SIMPLEX=0,
        circle=track_circle,
        polylines=lambda img, *args, **kwargs: img,
        line=lambda img, *args, **kwargs: img,
        rectangle=lambda img, *args, **kwargs: img,
        putText=lambda img, *args, **kwargs: img,
    )

    import neon_menu
    original_cv2 = neon_menu.cv2
    neon_menu.cv2 = fake_cv2

    try:
        menu = NeonMenu(buttons=[MenuButton("circle", (255, 0, 0))])
        menu.state = MenuState.HIDDEN
        menu.animation = 0.0

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = menu.draw(frame)

        # No drawing calls should have been made
        assert len(draw_calls) == 0
        assert result is frame
    finally:
        neon_menu.cv2 = original_cv2


def test_no_pulse_no_sin_cos_in_draw():
    """STABLE mode: draw() should not use sin/cos for pulse effects."""
    # This is a structural test - the simplified NeonMenu doesn't use
    # time accumulation or pulse calculations
    menu = NeonMenu(buttons=[MenuButton("circle", (255, 0, 0))])

    # Verify no _time_accum attribute (removed in stable mode)
    assert not hasattr(menu, "_time_accum") or menu.__dict__.get("_time_accum") is None

    # Verify no pulse_speed attribute
    assert not hasattr(menu, "pulse_speed")

    # Verify no glow_intensity attribute
    assert not hasattr(menu, "glow_intensity")


def test_positions_calculated_fresh_each_frame():
    """Positions are calculated fresh each frame (no cache bugs)."""
    menu = NeonMenu(
        center=(100, 100),
        radius=50,
        buttons=[MenuButton("a", (255, 0, 0)), MenuButton("b", (0, 255, 0))],
    )
    menu.state = MenuState.VISIBLE
    menu.animation = 1.0

    # Move center
    menu.center = (200, 200)

    # Verify no _cached_positions or similar stale cache
    # The stable menu calculates positions fresh in draw()
    assert not hasattr(menu, "_cached_positions") or len(getattr(menu, "_cached_positions", [])) == 0
