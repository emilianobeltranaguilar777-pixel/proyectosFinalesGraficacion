"""
Smoke tests for AR filter module.

These tests verify:
- Module imports work correctly
- No OpenGL dependency in metrics/primitives
- Basic functionality without camera/display
"""

import pytest


class TestModuleImports:
    """Test that all AR filter modules can be imported."""

    def test_import_ar_filter_package(self):
        """AR filter package should be importable."""
        import ar_filter
        assert hasattr(ar_filter, '__version__')

    def test_import_metrics(self):
        """Metrics module should import without OpenGL."""
        from ar_filter import metrics
        assert hasattr(metrics, 'clamp')
        assert hasattr(metrics, 'face_width')
        assert hasattr(metrics, 'halo_radius')
        assert hasattr(metrics, 'mouth_openness')

    def test_import_primitives(self):
        """Primitives module should import without OpenGL."""
        from ar_filter import primitives
        assert hasattr(primitives, 'build_sphere')
        assert hasattr(primitives, 'build_quad')
        assert hasattr(primitives, 'build_icosphere')

    def test_import_face_tracker(self):
        """Face tracker module should be importable."""
        from ar_filter import face_tracker
        assert hasattr(face_tracker, 'FaceTracker')
        assert hasattr(face_tracker, 'FaceLandmarks')

    def test_import_gl_app(self):
        """GL app module should be importable (may fail without display)."""
        try:
            from ar_filter import gl_app
            assert hasattr(gl_app, 'ARFilterApp')
            assert hasattr(gl_app, 'run_ar_filter')
        except ImportError:
            # Expected if OpenGL/GLFW not available
            pytest.skip("OpenGL/GLFW not available")


class TestMetricsNoOpenGL:
    """Verify metrics module has no OpenGL dependencies."""

    def test_metrics_uses_only_math(self):
        """Metrics should only use standard library math."""
        from ar_filter import metrics
        import inspect

        source = inspect.getsource(metrics)

        # Should NOT contain OpenGL imports (check actual import statements)
        assert 'import OpenGL' not in source
        assert 'from OpenGL' not in source
        assert 'import glfw' not in source
        assert 'from glfw' not in source

        # Should contain math imports
        assert 'import math' in source or 'from math' in source

    def test_metrics_functions_are_pure(self):
        """Metrics functions should be callable with just data."""
        from ar_filter.metrics import (
            clamp, face_width, halo_radius, mouth_openness,
            lerp, smooth_value, mouth_rect_scale, mouth_rect_color
        )

        # All should work with simple inputs
        assert clamp(5, 0, 10) == 5
        assert halo_radius(0.3) > 0
        assert lerp(0, 10, 0.5) == 5
        assert smooth_value(0, 10, 0.5) == 5

        # With mock landmarks
        mock_landmarks = [(0.5, 0.5, 0.0)] * 500
        assert face_width(mock_landmarks) >= 0
        assert mouth_openness(mock_landmarks) >= 0


class TestPrimitivesNoOpenGL:
    """Verify primitives module has no OpenGL dependencies."""

    def test_primitives_uses_only_numpy(self):
        """Primitives should only use numpy for geometry."""
        from ar_filter import primitives
        import inspect

        source = inspect.getsource(primitives)

        # Should NOT contain OpenGL imports (check actual import statements)
        assert 'import OpenGL' not in source
        assert 'from OpenGL' not in source
        assert 'import glfw' not in source
        assert 'from glfw' not in source

        # Should use numpy
        assert 'numpy' in source or 'np' in source

    def test_primitives_return_numpy_arrays(self):
        """Primitives should return numpy arrays."""
        import numpy as np
        from ar_filter.primitives import build_sphere, build_quad, build_icosphere

        # Sphere
        verts, norms, indices = build_sphere()
        assert isinstance(verts, np.ndarray)
        assert isinstance(norms, np.ndarray)
        assert isinstance(indices, np.ndarray)

        # Quad
        verts, norms, indices = build_quad()
        assert isinstance(verts, np.ndarray)
        assert isinstance(norms, np.ndarray)
        assert isinstance(indices, np.ndarray)

        # Icosphere
        verts, norms, indices = build_icosphere()
        assert isinstance(verts, np.ndarray)
        assert isinstance(norms, np.ndarray)
        assert isinstance(indices, np.ndarray)


class TestFaceTrackerSafe:
    """Test face tracker in safe mode (without MediaPipe if unavailable)."""

    def test_face_landmarks_constants(self):
        """FaceLandmarks should have expected constants."""
        from ar_filter.face_tracker import FaceLandmarks

        assert FaceLandmarks.NOSE_TIP == 1
        assert FaceLandmarks.UPPER_LIP_CENTER == 13
        assert FaceLandmarks.LOWER_LIP_CENTER == 14
        assert FaceLandmarks.LEFT_CHEEK == 234
        assert FaceLandmarks.RIGHT_CHEEK == 454

    def test_landmarks_to_screen(self):
        """landmarks_to_screen should convert normalized to pixel coords."""
        from ar_filter.face_tracker import landmarks_to_screen

        landmarks = [(0.5, 0.5, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        width, height = 640, 480

        result = landmarks_to_screen(landmarks, width, height)

        assert len(result) == 3
        assert result[0] == (320, 240)  # Center
        assert result[1] == (0, 0)      # Top-left
        assert result[2] == (640, 480)  # Bottom-right

    def test_landmarks_to_screen_with_indices(self):
        """landmarks_to_screen should respect index selection."""
        from ar_filter.face_tracker import landmarks_to_screen

        landmarks = [(0.25, 0.25, 0.0), (0.5, 0.5, 0.0), (0.75, 0.75, 0.0)]

        result = landmarks_to_screen(landmarks, 100, 100, indices=[1])

        assert len(result) == 1
        assert result[0] == (50, 50)


class TestShaderFiles:
    """Test that shader files exist and are valid."""

    def test_vertex_shader_exists(self):
        """Vertex shader file should exist."""
        import os
        shader_path = os.path.join(
            os.path.dirname(__file__), '..', 'ar_filter', 'shaders', 'basic.vert'
        )
        assert os.path.exists(shader_path), f"Missing: {shader_path}"

    def test_fragment_shader_exists(self):
        """Fragment shader file should exist."""
        import os
        shader_path = os.path.join(
            os.path.dirname(__file__), '..', 'ar_filter', 'shaders', 'basic.frag'
        )
        assert os.path.exists(shader_path), f"Missing: {shader_path}"

    def test_vertex_shader_has_version(self):
        """Vertex shader should declare GLSL version."""
        import os
        shader_path = os.path.join(
            os.path.dirname(__file__), '..', 'ar_filter', 'shaders', 'basic.vert'
        )
        with open(shader_path, 'r') as f:
            content = f.read()

        assert '#version' in content
        assert '150' in content  # OpenGL 3.2 compatible

    def test_fragment_shader_has_version(self):
        """Fragment shader should declare GLSL version."""
        import os
        shader_path = os.path.join(
            os.path.dirname(__file__), '..', 'ar_filter', 'shaders', 'basic.frag'
        )
        with open(shader_path, 'r') as f:
            content = f.read()

        assert '#version' in content
        assert '150' in content  # OpenGL 3.2 compatible


class TestIntegrationPoints:
    """Test that integration points are correctly set up."""

    def test_ar_filter_entry_point_exists(self):
        """run_ar_filter function should be importable."""
        try:
            from ar_filter.gl_app import run_ar_filter
            assert callable(run_ar_filter)
        except ImportError:
            pytest.skip("OpenGL not available")

    def test_ar_filter_app_class_exists(self):
        """ARFilterApp class should be importable."""
        try:
            from ar_filter.gl_app import ARFilterApp
            assert ARFilterApp is not None
        except ImportError:
            pytest.skip("OpenGL not available")
