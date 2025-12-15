import importlib


def test_module_import_without_opengl():
    module = importlib.import_module("ar_filter")
    assert hasattr(module, "run_ar_filter")


def test_gl_app_instantiation_headless():
    from ar_filter.gl_app import ARFilterApp

    app = ARFilterApp(width=320, height=240)
    # Only ensure core attributes are initialized without creating GL context
    assert app.width == 320
    assert app.height == 240
