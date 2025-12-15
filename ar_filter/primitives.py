"""Geometry helpers for simple AR overlay rendering."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def create_sphere(radius: float = 1.0, segments: int = 8, rings: int = 8) -> List[Tuple[float, float, float]]:
    """Generates a simple sphere mesh as a list of vertices."""
    segments = max(segments, 3)
    rings = max(rings, 3)
    vertices = []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(segments + 1):
            theta = 2.0 * math.pi * s / segments
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.cos(phi)
            z = radius * math.sin(phi) * math.sin(theta)
            vertices.append((x, y, z))
    return vertices


def create_quad(center: Sequence[float] = (0.0, 0.0, 0.0), size: Sequence[float] = (1.0, 1.0)) -> List[Tuple[float, float, float]]:
    """Returns four vertices representing a quad in the X/Y plane."""
    cx, cy, cz = center
    sx, sy = size[0] * 0.5, size[1] * 0.5
    return [
        (cx - sx, cy - sy, cz),
        (cx + sx, cy - sy, cz),
        (cx + sx, cy + sy, cz),
        (cx - sx, cy + sy, cz),
    ]
