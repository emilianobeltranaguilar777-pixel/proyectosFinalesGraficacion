"""Tests for Angular Ring Scale Mode.

These tests verify the angular scaling behavior:
- Size increases with positive (counter-clockwise) angle delta
- Size decreases with negative (clockwise) angle delta
- Scaling is incremental, not absolute
- Size is clamped to min/max bounds
- State resets correctly on exit
- Scaling does not affect rotation
"""
import math
import sys
import types

import numpy as np
import pytest


def _install_fake_cv2():
    """Install minimal cv2 stub for headless testing."""
    fake_cv2 = types.SimpleNamespace(
        LINE_AA=1,
        FONT_HERSHEY_SIMPLEX=0,
        WINDOW_NORMAL=0,
        COLOR_BGR2RGB=0,
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
    fake_cv2.addWeighted = lambda src1, alpha, src2, beta, gamma, dst=None: src1
    fake_cv2.resize = lambda img, size, interpolation=None: np.zeros((size[1], size[0], 3), dtype=np.uint8)
    fake_cv2.flip = _return_img

    sys.modules["cv2"] = fake_cv2


# Install fake cv2 before importing Gesture3D
_install_fake_cv2()

from Gesture3D import Gesture3D, SelectionMode


@pytest.fixture
def gesture3d(monkeypatch):
    """Create Gesture3D instance without MediaPipe."""
    monkeypatch.setattr(Gesture3D, "_initialize_mediapipe", lambda self: None)
    g3d = Gesture3D(640, 480, use_external_menu=False)
    g3d.mediapipe_available = False
    return g3d


@pytest.fixture
def gesture3d_with_figure(gesture3d):
    """Create Gesture3D with a selected figure in scale mode."""
    gesture3d.create_figure('circle', (320, 240))
    gesture3d.selection_mode = SelectionMode.SCALE
    gesture3d.selected_figure['size'] = 100  # Known starting size
    return gesture3d


class TestAngularScaleIncrements:
    """Test that angular scaling is incremental and continuous."""

    def test_scale_increases_with_positive_angle_delta(self, gesture3d_with_figure):
        """Counter-clockwise rotation (positive delta) should increase size."""
        g3d = gesture3d_with_figure
        initial_size = g3d.selected_figure['size']
        fig_center = g3d.selected_figure['position']

        # First call: establish initial angle (at 0 degrees, right side)
        pinch_pos_1 = (fig_center[0] + 100, fig_center[1])  # angle = 0
        g3d.handle_figure_scaling_by_angular(pinch_pos_1)

        # Size should not change on first call (just recording angle)
        assert g3d.selected_figure['size'] == initial_size

        # Second call: move counter-clockwise (positive angle delta)
        # Move from 0 to ~45 degrees (π/4 radians)
        offset = int(100 * math.cos(math.pi / 4))
        pinch_pos_2 = (fig_center[0] + offset, fig_center[1] - offset)  # angle ≈ -π/4
        g3d.handle_figure_scaling_by_angular(pinch_pos_2)

        # Size should increase
        assert g3d.selected_figure['size'] > initial_size

    def test_scale_decreases_with_negative_angle_delta(self, gesture3d_with_figure):
        """Clockwise rotation (negative delta) should decrease size."""
        g3d = gesture3d_with_figure
        initial_size = g3d.selected_figure['size']
        fig_center = g3d.selected_figure['position']

        # First call: establish initial angle at 0 degrees
        pinch_pos_1 = (fig_center[0] + 100, fig_center[1])  # angle = 0
        g3d.handle_figure_scaling_by_angular(pinch_pos_1)

        # Second call: move clockwise (negative angle delta)
        # Move from 0 to ~-45 degrees
        offset = int(100 * math.cos(math.pi / 4))
        pinch_pos_2 = (fig_center[0] + offset, fig_center[1] + offset)  # angle ≈ π/4
        g3d.handle_figure_scaling_by_angular(pinch_pos_2)

        # Size should decrease
        assert g3d.selected_figure['size'] < initial_size

    def test_scale_is_incremental_not_absolute(self, gesture3d_with_figure):
        """Each angular change should add to current size, not set absolute."""
        g3d = gesture3d_with_figure
        fig_center = g3d.selected_figure['position']

        # Set initial size
        g3d.selected_figure['size'] = 100

        # First call: establish angle
        g3d.handle_figure_scaling_by_angular((fig_center[0] + 100, fig_center[1]))

        # Apply small incremental rotations
        sizes = [g3d.selected_figure['size']]
        radius = 100

        # Rotate counter-clockwise in small steps (negative angle in screen coords = up)
        for i in range(1, 5):
            angle = -i * 0.1  # Negative = counter-clockwise in screen coords = grow
            x = int(fig_center[0] + radius * math.cos(angle))
            y = int(fig_center[1] + radius * math.sin(angle))
            g3d.handle_figure_scaling_by_angular((x, y))
            sizes.append(g3d.selected_figure['size'])

        # Each step should show incremental change
        # All sizes after first should be larger (counter-clockwise in screen = grow)
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1], f"Size should increase: {sizes}"


class TestAngularScaleBounds:
    """Test that scaling respects min/max bounds."""

    def test_scale_clamped_to_min_max(self, gesture3d_with_figure):
        """Size should not exceed min_figure_size or max_figure_size."""
        g3d = gesture3d_with_figure
        fig_center = g3d.selected_figure['position']

        # Test max bound
        g3d.selected_figure['size'] = g3d.max_figure_size - 10

        # Establish angle
        g3d.handle_figure_scaling_by_angular((fig_center[0] + 100, fig_center[1]))

        # Try to scale way up (full rotation counter-clockwise)
        for i in range(20):
            angle = -i * 0.5  # Large counter-clockwise rotation
            x = int(fig_center[0] + 100 * math.cos(angle))
            y = int(fig_center[1] + 100 * math.sin(angle))
            g3d.handle_figure_scaling_by_angular((x, y))

        assert g3d.selected_figure['size'] <= g3d.max_figure_size

        # Reset for min bound test
        g3d.reset_angular_scale_state()
        g3d.selected_figure['size'] = g3d.min_figure_size + 10

        # Establish angle
        g3d.handle_figure_scaling_by_angular((fig_center[0] + 100, fig_center[1]))

        # Try to scale way down (clockwise rotation)
        for i in range(20):
            angle = i * 0.5  # Large clockwise rotation
            x = int(fig_center[0] + 100 * math.cos(angle))
            y = int(fig_center[1] + 100 * math.sin(angle))
            g3d.handle_figure_scaling_by_angular((x, y))

        assert g3d.selected_figure['size'] >= g3d.min_figure_size


class TestAngularScaleStateManagement:
    """Test state management for angular scaling."""

    def test_scale_state_resets_on_exit(self, gesture3d_with_figure):
        """State should reset when exiting scale mode."""
        g3d = gesture3d_with_figure
        fig_center = g3d.selected_figure['position']

        # Start scaling
        g3d.handle_figure_scaling_by_angular((fig_center[0] + 100, fig_center[1]))
        assert g3d._scale_prev_angle is not None
        assert g3d._scale_angular_active is True

        # Exit scale mode
        g3d.toggle_scale_mode()

        # State should be reset
        assert g3d._scale_prev_angle is None
        assert g3d._scale_angular_active is False
        assert g3d.selection_mode == SelectionMode.NORMAL

    def test_scale_state_resets_on_pinch_release(self, gesture3d_with_figure):
        """State should reset when pinch is released."""
        g3d = gesture3d_with_figure
        fig_center = g3d.selected_figure['position']

        # Start scaling
        g3d._scale_prev_angle = 0.5
        g3d._scale_angular_active = True

        # Simulate pinch release
        g3d.reset_angular_scale_state()

        assert g3d._scale_prev_angle is None
        assert g3d._scale_angular_active is False

    def test_first_frame_records_angle_no_scale(self, gesture3d_with_figure):
        """First scaling call should only record angle, not change size."""
        g3d = gesture3d_with_figure
        initial_size = g3d.selected_figure['size']
        fig_center = g3d.selected_figure['position']

        # First call
        g3d.handle_figure_scaling_by_angular((fig_center[0] + 100, fig_center[1]))

        # Size should be unchanged
        assert g3d.selected_figure['size'] == initial_size
        # But angle should be recorded
        assert g3d._scale_prev_angle is not None


class TestAngularScaleIsolation:
    """Test that scaling does not affect other features."""

    def test_scale_does_not_affect_rotation(self, gesture3d_with_figure):
        """Scaling should not change figure rotation."""
        g3d = gesture3d_with_figure
        g3d.selected_figure['rotation'] = 1.5  # Set known rotation
        initial_rotation = g3d.selected_figure['rotation']
        fig_center = g3d.selected_figure['position']

        # Perform scaling
        g3d.handle_figure_scaling_by_angular((fig_center[0] + 100, fig_center[1]))
        g3d.handle_figure_scaling_by_angular((fig_center[0], fig_center[1] - 100))
        g3d.handle_figure_scaling_by_angular((fig_center[0] - 100, fig_center[1]))

        # Rotation should be unchanged
        assert g3d.selected_figure['rotation'] == initial_rotation

    def test_scale_does_not_affect_position(self, gesture3d_with_figure):
        """Scaling should not change figure position."""
        g3d = gesture3d_with_figure
        initial_position = g3d.selected_figure['position']
        fig_center = g3d.selected_figure['position']

        # Perform scaling
        g3d.handle_figure_scaling_by_angular((fig_center[0] + 100, fig_center[1]))
        g3d.handle_figure_scaling_by_angular((fig_center[0], fig_center[1] - 100))

        # Position should be unchanged
        assert g3d.selected_figure['position'] == initial_position

    def test_scale_requires_selected_figure(self, gesture3d):
        """Scaling should do nothing without selected figure."""
        g3d = gesture3d
        g3d.selection_mode = SelectionMode.SCALE
        g3d.selected_figure = None

        # Should not raise, just return
        g3d.handle_figure_scaling_by_angular((100, 100))

        # No state change
        assert g3d._scale_prev_angle is None

    def test_scale_handles_pinch_at_center(self, gesture3d_with_figure):
        """Scaling should handle pinch at figure center gracefully."""
        g3d = gesture3d_with_figure
        initial_size = g3d.selected_figure['size']
        fig_center = g3d.selected_figure['position']

        # Pinch exactly at center
        g3d.handle_figure_scaling_by_angular(fig_center)

        # Size should not change (avoided division by zero)
        assert g3d.selected_figure['size'] == initial_size


class TestAngularScaleWrapAround:
    """Test handling of angle wrap-around at ±π."""

    def test_scale_handles_angle_wraparound(self, gesture3d_with_figure):
        """Crossing ±π boundary should not cause sudden large changes."""
        g3d = gesture3d_with_figure
        fig_center = g3d.selected_figure['position']

        # Start near π
        angle_start = math.pi - 0.1
        x1 = int(fig_center[0] + 100 * math.cos(angle_start))
        y1 = int(fig_center[1] + 100 * math.sin(angle_start))
        g3d.handle_figure_scaling_by_angular((x1, y1))

        size_before = g3d.selected_figure['size']

        # Cross to -π + small offset (should be small angular change)
        angle_end = -math.pi + 0.1
        x2 = int(fig_center[0] + 100 * math.cos(angle_end))
        y2 = int(fig_center[1] + 100 * math.sin(angle_end))
        g3d.handle_figure_scaling_by_angular((x2, y2))

        size_after = g3d.selected_figure['size']

        # Change should be small (not a full 2π jump)
        # The actual delta is ~0.2 radians, so size change should be modest
        size_change = abs(size_after - size_before)
        assert size_change < 50, f"Size changed too much during wraparound: {size_change}"
