"""AR filter module providing simple OpenGL overlay effects."""

from importlib import import_module
from typing import Any

__all__ = [
    "mouth_open_ratio",
    "face_width",
    "halo_radius",
    "smooth_value",
    "create_sphere",
    "create_quad",
    "FaceTracker",
    "ARFilterApp",
    "run_ar_filter",
]


def _load(module: str, name: str) -> Any:
    return getattr(import_module(module, __name__), name)


def mouth_open_ratio(landmarks):
    return _load(".metrics", "mouth_open_ratio")(landmarks)


def face_width(landmarks):
    return _load(".metrics", "face_width")(landmarks)


def halo_radius(landmarks, scale: float = 1.1):
    return _load(".metrics", "halo_radius")(landmarks, scale=scale)


def smooth_value(current: float, target: float, factor: float):
    return _load(".metrics", "smooth_value")(current, target, factor)


def create_sphere(radius: float = 1.0, segments: int = 8, rings: int = 8):
    return _load(".primitives", "create_sphere")(radius=radius, segments=segments, rings=rings)


def create_quad(center=(0.0, 0.0, 0.0), size=(1.0, 1.0)):
    return _load(".primitives", "create_quad")(center=center, size=size)


class FaceTracker:  # type: ignore[override]
    def __new__(cls, *args, **kwargs):
        tracker_cls = _load(".face_tracker", "FaceTracker")
        return tracker_cls(*args, **kwargs)


class ARFilterApp:  # type: ignore[override]
    def __new__(cls, *args, **kwargs):
        app_cls = _load(".gl_app", "ARFilterApp")
        return app_cls(*args, **kwargs)


def run_ar_filter():
    return _load(".gl_app", "run_ar_filter")()
