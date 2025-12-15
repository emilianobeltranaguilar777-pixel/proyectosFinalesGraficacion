"""
GL Safety Tests for AR Filter Module

These tests ensure that the AR filter does NOT use deprecated or
problematic OpenGL APIs that fail on Core Profile (Ubuntu).

If any of these tests fail, the CI should block the commit.
"""

import pytest
import inspect


class TestNoGLLineWidth:
    """Tests to ensure glLineWidth is NEVER used."""

    def test_no_glLineWidth_in_gl_app(self):
        """gl_app.py must NOT contain glLineWidth function calls."""
        import ar_filter.gl_app as gl_app
        import re
        source = inspect.getsource(gl_app)

        # Check for actual glLineWidth calls (not comments)
        # Match: glLineWidth( with optional spaces before (
        pattern = r'[^#]*\bglLineWidth\s*\('
        matches = re.findall(pattern, source)

        assert len(matches) == 0, \
            f"FORBIDDEN: glLineWidth call found in gl_app.py - causes GL_INVALID_VALUE in Core Profile!"

    def test_no_glLineWidth_in_primitives(self):
        """primitives.py must NOT contain any OpenGL calls."""
        import ar_filter.primitives as primitives
        source = inspect.getsource(primitives)

        # Should not have any GL calls at all
        assert "glLineWidth" not in source, \
            "FORBIDDEN: glLineWidth found in primitives.py!"
        assert "from OpenGL" not in source, \
            "FORBIDDEN: OpenGL imports found in primitives.py - it should be GL-free!"


class TestNoDeprecatedGLModes:
    """Tests to ensure deprecated GL draw modes are not used."""

    def test_no_GL_LINE_STRIP_usage(self):
        """gl_app.py should not use GL_LINE_STRIP."""
        import ar_filter.gl_app as gl_app
        source = inspect.getsource(gl_app)

        # Check that GL_LINE_STRIP is not used in draw calls
        # It can be imported but shouldn't be used
        assert "glDrawArrays(GL_LINE_STRIP" not in source, \
            "FORBIDDEN: GL_LINE_STRIP used in glDrawArrays - use GL_TRIANGLES instead!"

    def test_no_GL_LINE_LOOP_usage(self):
        """gl_app.py should not use GL_LINE_LOOP."""
        import ar_filter.gl_app as gl_app
        source = inspect.getsource(gl_app)

        assert "glDrawArrays(GL_LINE_LOOP" not in source, \
            "FORBIDDEN: GL_LINE_LOOP used in glDrawArrays - use GL_TRIANGLES instead!"

    def test_no_GL_LINES_usage(self):
        """gl_app.py should not use GL_LINES."""
        import ar_filter.gl_app as gl_app
        source = inspect.getsource(gl_app)

        assert "glDrawArrays(GL_LINES" not in source, \
            "FORBIDDEN: GL_LINES used in glDrawArrays - use GL_TRIANGLES instead!"


class TestCoreProfileSafeAPIs:
    """Tests for Core Profile compatible API usage."""

    def test_uses_GL_TRIANGLES(self):
        """gl_app.py should use GL_TRIANGLES for drawing."""
        import ar_filter.gl_app as gl_app
        source = inspect.getsource(gl_app)

        assert "GL_TRIANGLES" in source, \
            "gl_app.py should use GL_TRIANGLES for all drawing!"

    def test_has_draw_triangles_method(self):
        """NeonMaskFilter should have _draw_triangles method."""
        from ar_filter.gl_app import NeonMaskFilter

        assert hasattr(NeonMaskFilter, '_draw_triangles'), \
            "NeonMaskFilter must have _draw_triangles method!"

    def test_no_draw_vertices_with_lines(self):
        """Should not have a generic _draw_vertices that accepts line modes."""
        import ar_filter.gl_app as gl_app
        source = inspect.getsource(gl_app)

        # Old method signature that accepted mode parameter
        # Should not exist or should be removed
        assert "_draw_vertices(self, vertices: np.ndarray, mode: int" not in source, \
            "FORBIDDEN: _draw_vertices with mode parameter found - use _draw_triangles!"


class TestViewportSetup:
    """Tests for proper viewport setup."""

    def test_glViewport_called_in_render(self):
        """gl_app.py should call glViewport before rendering."""
        import ar_filter.gl_app as gl_app
        source = inspect.getsource(gl_app)

        assert "glViewport" in source, \
            "glViewport must be called before rendering!"

    def test_glClear_called(self):
        """gl_app.py should call glClear."""
        import ar_filter.gl_app as gl_app
        source = inspect.getsource(gl_app)

        assert "glClear" in source, \
            "glClear must be called to clear the framebuffer!"


class TestBlendingSetup:
    """Tests for proper blending setup."""

    def test_blend_enabled(self):
        """gl_app.py should enable blending."""
        import ar_filter.gl_app as gl_app
        source = inspect.getsource(gl_app)

        assert "glEnable(GL_BLEND)" in source, \
            "GL_BLEND must be enabled for transparency!"

    def test_blend_func_set(self):
        """gl_app.py should set blend function."""
        import ar_filter.gl_app as gl_app
        source = inspect.getsource(gl_app)

        assert "glBlendFunc" in source, \
            "glBlendFunc must be called to set blend mode!"


class TestNoForbiddenStates:
    """Tests for absence of problematic GL states."""

    def test_no_depth_test(self):
        """gl_app.py should NOT use depth testing for 2D overlay."""
        import ar_filter.gl_app as gl_app
        source = inspect.getsource(gl_app)

        # Depth testing is unnecessary for 2D and can cause issues
        assert "glEnable(GL_DEPTH_TEST)" not in source, \
            "GL_DEPTH_TEST should not be enabled for 2D rendering!"

    def test_no_stencil_test(self):
        """gl_app.py should NOT use stencil testing."""
        import ar_filter.gl_app as gl_app
        source = inspect.getsource(gl_app)

        assert "glEnable(GL_STENCIL_TEST)" not in source, \
            "GL_STENCIL_TEST should not be used - keep it simple!"


class TestPrimitivesReturnTriangles:
    """Tests that primitives return triangle-based geometry."""

    def test_build_circle_returns_triangles(self):
        """build_circle should return triangle vertices."""
        from ar_filter.primitives import build_circle

        circle = build_circle((0.5, 0.5), 0.1)

        # Triangle-based circle has 6 vertices per segment (2 triangles)
        # With 32 segments default, that's 192 vertices
        assert len(circle) >= 6, \
            "build_circle should return triangle vertices (at least 6)"
        assert len(circle) % 6 == 0, \
            "build_circle vertex count should be multiple of 6 (2 triangles per segment)"

    def test_build_star_returns_triangles(self):
        """build_star should return triangle vertices."""
        from ar_filter.primitives import build_star

        star = build_star((0.5, 0.5), 0.1)

        # Triangle-based star has many vertices
        assert len(star) >= 6, \
            "build_star should return triangle vertices"

    def test_build_horn_returns_triangles(self):
        """build_horn should return triangle vertices."""
        from ar_filter.primitives import build_horn

        horn = build_horn((0.5, 0.5), 1.0, 0.0)

        # Triangle-based horn strip
        assert len(horn) >= 6, \
            "build_horn should return triangle vertices"


class TestRenderContract:
    """Tests for the rendering contract."""

    def test_render_method_exists(self):
        """NeonMaskFilter should have run method."""
        from ar_filter.gl_app import NeonMaskFilter

        assert hasattr(NeonMaskFilter, 'run'), \
            "NeonMaskFilter must have run() method!"

    def test_cleanup_method_exists(self):
        """NeonMaskFilter should have cleanup method."""
        from ar_filter.gl_app import NeonMaskFilter

        assert hasattr(NeonMaskFilter, 'cleanup'), \
            "NeonMaskFilter must have cleanup() method!"

    def test_render_neon_mask_exists(self):
        """NeonMaskFilter should have _render_neon_mask method."""
        from ar_filter.gl_app import NeonMaskFilter

        assert hasattr(NeonMaskFilter, '_render_neon_mask'), \
            "NeonMaskFilter must have _render_neon_mask() method!"
