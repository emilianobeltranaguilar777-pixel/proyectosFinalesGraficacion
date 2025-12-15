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


def test_ar_filter_fails_cleanly_without_camera(monkeypatch):
    import sys
    import types

    from ar_filter.gl_app import ARFilterApp

    captured_cap = {}
    window_called = {"called": False}

    def fake_init_window(self):
        window_called["called"] = True
        raise AssertionError("_init_window should not be called when camera is unavailable")

    def fake_setup_buffers(self):
        raise AssertionError("Buffers should not be set up when camera is unavailable")

    class FakeCap:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    fake_cv2 = types.SimpleNamespace(CAP_V4L2=0)

    def fake_video_capture(*args, **kwargs):
        cap = FakeCap()
        captured_cap["cap"] = cap
        return cap

    fake_cv2.VideoCapture = fake_video_capture
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(ARFilterApp, "_init_window", fake_init_window, raising=False)
    monkeypatch.setattr(ARFilterApp, "_setup_buffers", fake_setup_buffers, raising=False)

    app = ARFilterApp()
    app.run()

    assert captured_cap["cap"].released is True
    assert window_called["called"] is False
