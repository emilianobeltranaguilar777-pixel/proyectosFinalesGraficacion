"""
Smoke Tests for AR Filter Module

Tests that modules can be imported and basic functionality works.
Does NOT require OpenGL, camera, or display.
All tests can run headless.
"""

import pytest
import sys


class TestModuleImports:
    """Tests for module import availability."""

    def test_import_ar_filter_package(self):
        """ar_filter package should be importable."""
        import ar_filter
        assert ar_filter is not None

    def test_import_face_tracker(self):
        """face_tracker module should be importable."""
        from ar_filter import face_tracker
        assert face_tracker is not None

    def test_import_metrics(self):
        """metrics module should be importable."""
        from ar_filter import metrics
        assert metrics is not None

    def test_import_primitives(self):
        """primitives module should be importable."""
        from ar_filter import primitives
        assert primitives is not None

    def test_import_gl_app(self):
        """gl_app module should be importable (even without OpenGL)."""
        from ar_filter import gl_app
        assert gl_app is not None


class TestFaceTrackerConstruction:
    """Tests for FaceTracker class construction."""

    def test_create_face_tracker(self):
        """FaceTracker should be constructible."""
        from ar_filter.face_tracker import FaceTracker
        tracker = FaceTracker()
        assert tracker is not None

    def test_face_tracker_attributes(self):
        """FaceTracker should have expected attributes."""
        from ar_filter.face_tracker import FaceTracker
        tracker = FaceTracker()

        assert hasattr(tracker, 'max_faces')
        assert hasattr(tracker, 'min_detection_confidence')
        assert hasattr(tracker, 'is_available')

    def test_face_tracker_methods(self):
        """FaceTracker should have expected methods."""
        from ar_filter.face_tracker import FaceTracker
        tracker = FaceTracker()

        assert callable(getattr(tracker, 'process_frame', None))
        assert callable(getattr(tracker, 'get_key_points', None))
        assert callable(getattr(tracker, 'release', None))

    def test_face_tracker_landmark_indices(self):
        """FaceTracker should define landmark indices."""
        from ar_filter.face_tracker import FaceTracker

        assert FaceTracker.NOSE_TIP == 1
        assert FaceTracker.CHIN == 152
        assert FaceTracker.LEFT_EYE_OUTER == 33
        assert FaceTracker.RIGHT_EYE_OUTER == 263


class TestMetricsFunctions:
    """Tests for metrics function availability."""

    def test_metrics_functions_exist(self):
        """All expected metrics functions should exist."""
        from ar_filter.metrics import (
            face_width,
            face_height,
            mouth_openness,
            head_tilt,
            head_rotation_y,
            face_center,
            eye_center,
            eyebrow_raise,
            smile_intensity,
            interpolate_position,
            smooth_value,
        )

        assert callable(face_width)
        assert callable(face_height)
        assert callable(mouth_openness)
        assert callable(head_tilt)
        assert callable(face_center)
        assert callable(smooth_value)


class TestPrimitivesFunctions:
    """Tests for primitives function availability."""

    def test_primitives_functions_exist(self):
        """All expected primitives functions should exist."""
        from ar_filter.primitives import (
            build_circle,
            build_filled_circle,
            build_horn,
            build_mask_outline,
            build_cat_ears,
            build_halo,
            build_neon_lines,
            build_zigzag_line,
            build_star,
            build_heart,
        )

        assert callable(build_circle)
        assert callable(build_horn)
        assert callable(build_star)
        assert callable(build_heart)


class TestGlAppComponents:
    """Tests for gl_app components (without OpenGL)."""

    def test_neon_mask_filter_class_exists(self):
        """NeonMaskFilter class should exist."""
        from ar_filter.gl_app import NeonMaskFilter
        assert NeonMaskFilter is not None

    def test_run_ar_filter_exists(self):
        """run_ar_filter function should exist."""
        from ar_filter.gl_app import run_ar_filter
        assert callable(run_ar_filter)

    def test_opengl_available_flag(self):
        """OPENGL_AVAILABLE flag should be defined."""
        from ar_filter.gl_app import OPENGL_AVAILABLE
        assert isinstance(OPENGL_AVAILABLE, bool)

    def test_neon_mask_colors_defined(self):
        """NeonMaskFilter should have COLORS defined."""
        from ar_filter.gl_app import NeonMaskFilter

        assert hasattr(NeonMaskFilter, 'COLORS')
        assert 'cyan' in NeonMaskFilter.COLORS
        assert 'magenta' in NeonMaskFilter.COLORS


class TestNeonMaskFilterConstruction:
    """Tests for NeonMaskFilter construction (without initialization)."""

    def test_create_filter_object(self):
        """NeonMaskFilter should be constructible without initialization."""
        from ar_filter.gl_app import NeonMaskFilter

        # Just construct, don't initialize (would need OpenGL)
        filter_app = NeonMaskFilter(width=640, height=480)

        assert filter_app is not None
        assert filter_app.width == 640
        assert filter_app.height == 480

    def test_filter_has_face_tracker(self):
        """NeonMaskFilter should create a FaceTracker."""
        from ar_filter.gl_app import NeonMaskFilter

        filter_app = NeonMaskFilter()
        assert filter_app.face_tracker is not None

    def test_filter_animation_state(self):
        """NeonMaskFilter should have animation state variables."""
        from ar_filter.gl_app import NeonMaskFilter

        filter_app = NeonMaskFilter()

        assert hasattr(filter_app, 'smooth_mouth')
        assert hasattr(filter_app, 'smooth_tilt')
        assert hasattr(filter_app, 'current_color_idx')


class TestNoOpenGLRequired:
    """Tests that verify headless operation works."""

    def test_metrics_work_without_opengl(self):
        """Metrics should work without any OpenGL."""
        from ar_filter.metrics import face_width, mouth_openness

        landmarks = [(0.5, 0.5, 0.0)] * 468
        landmarks[234] = (0.3, 0.5, 0.0)
        landmarks[454] = (0.7, 0.5, 0.0)

        width = face_width(landmarks)
        assert width > 0

    def test_primitives_work_without_opengl(self):
        """Primitives should work without any OpenGL."""
        from ar_filter.primitives import build_circle, build_star

        circle = build_circle((0.5, 0.5), 0.1)
        star = build_star((0.5, 0.5), 0.2)

        assert len(circle) > 0
        assert len(star) > 0

    def test_face_tracker_construct_without_opengl(self):
        """FaceTracker should construct without OpenGL."""
        from ar_filter.face_tracker import FaceTracker

        tracker = FaceTracker()
        # Should not raise


class TestModuleVersion:
    """Tests for module metadata."""

    def test_version_defined(self):
        """Module should have __version__ defined."""
        import ar_filter
        assert hasattr(ar_filter, '__version__')
        assert ar_filter.__version__ == "1.0.0"


class TestIsolation:
    """Tests that verify module isolation from main project."""

    def test_no_gesture3d_import(self):
        """ar_filter should NOT import Gesture3D."""
        import ar_filter.gl_app

        # Check that Gesture3D is not in the module's namespace
        assert 'Gesture3D' not in dir(ar_filter.gl_app)

    def test_no_color_painter_import(self):
        """ar_filter should NOT import ColorPainter."""
        import ar_filter.gl_app

        assert 'ColorPainter' not in dir(ar_filter.gl_app)

    def test_no_neon_menu_import(self):
        """ar_filter should NOT import NeonMenu."""
        import ar_filter.gl_app

        assert 'NeonMenu' not in dir(ar_filter.gl_app)

    def test_no_main_import(self):
        """ar_filter should NOT import main."""
        import ar_filter.gl_app

        # 'main' module should not be loaded
        assert 'main' not in sys.modules or 'PizarraNeon' not in dir(ar_filter.gl_app)


class TestEntryPoint:
    """Tests for module entry point."""

    def test_run_ar_filter_exported(self):
        """run_ar_filter should be exported from package."""
        from ar_filter import run_ar_filter
        assert callable(run_ar_filter)

    def test_all_exports(self):
        """__all__ should be properly defined."""
        import ar_filter
        assert hasattr(ar_filter, '__all__')
        assert 'run_ar_filter' in ar_filter.__all__
