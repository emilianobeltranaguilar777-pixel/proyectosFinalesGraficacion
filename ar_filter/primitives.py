"""
Pure geometry generation for AR filter primitives.

This module generates vertex data for 3D shapes.
NO OpenGL imports here - just numpy arrays for geometry.
"""

import math
from typing import Tuple, List
import numpy as np


def build_sphere(radius: float = 1.0,
                 lat_segments: int = 8,
                 lon_segments: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a sphere using latitude-longitude tessellation.

    Args:
        radius: Sphere radius
        lat_segments: Number of latitude divisions (vertical)
        lon_segments: Number of longitude divisions (horizontal)

    Returns:
        Tuple of (vertices, normals, indices):
        - vertices: Nx3 array of vertex positions
        - normals: Nx3 array of vertex normals
        - indices: Mx3 array of triangle indices
    """
    vertices = []
    normals = []
    indices = []

    # Generate vertices
    for lat in range(lat_segments + 1):
        theta = (lat / lat_segments) * math.pi  # 0 to pi
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        for lon in range(lon_segments + 1):
            phi = (lon / lon_segments) * 2.0 * math.pi  # 0 to 2pi

            # Spherical to Cartesian
            x = sin_theta * math.cos(phi)
            y = cos_theta
            z = sin_theta * math.sin(phi)

            vertices.append([x * radius, y * radius, z * radius])
            normals.append([x, y, z])

    # Generate indices for triangles
    for lat in range(lat_segments):
        for lon in range(lon_segments):
            current = lat * (lon_segments + 1) + lon
            next_row = current + lon_segments + 1

            # Two triangles per quad
            indices.append([current, next_row, current + 1])
            indices.append([current + 1, next_row, next_row + 1])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(indices, dtype=np.uint32)
    )


def build_quad(width: float = 1.0,
               height: float = 1.0,
               center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a simple quad (two triangles) facing +Z.

    Args:
        width: Quad width
        height: Quad height
        center: Center position (x, y, z)

    Returns:
        Tuple of (vertices, normals, indices):
        - vertices: 4x3 array of vertex positions
        - normals: 4x3 array of vertex normals (all [0,0,1])
        - indices: 2x3 array of triangle indices
    """
    hw = width / 2.0
    hh = height / 2.0
    cx, cy, cz = center

    vertices = np.array([
        [cx - hw, cy - hh, cz],  # Bottom-left
        [cx + hw, cy - hh, cz],  # Bottom-right
        [cx + hw, cy + hh, cz],  # Top-right
        [cx - hw, cy + hh, cz],  # Top-left
    ], dtype=np.float32)

    normals = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    indices = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.uint32)

    return vertices, normals, indices


def build_icosphere(radius: float = 1.0,
                    subdivisions: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an icosphere (subdivision of icosahedron).

    More uniform distribution of vertices than lat-long sphere.
    Subdivisions: 0 = 12 vertices, 1 = 42 vertices, 2 = 162 vertices

    Args:
        radius: Sphere radius
        subdivisions: Number of subdivision iterations (0-3 recommended)

    Returns:
        Tuple of (vertices, normals, indices)
    """
    # Golden ratio
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    # Initial icosahedron vertices (normalized)
    vertices = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]

    # Normalize to unit sphere
    vertices = [[v[0] / math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for _ in range(1)] +
                [v[i] / math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for i in range(3)][1:]
                if False else
                [v[i] / math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for i in range(3)]
                for v in vertices]

    # Initial icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    # Subdivide
    vertex_cache = {}

    def get_middle_point(p1_idx: int, p2_idx: int) -> int:
        """Get or create middle point between two vertices."""
        key = (min(p1_idx, p2_idx), max(p1_idx, p2_idx))
        if key in vertex_cache:
            return vertex_cache[key]

        p1 = vertices[p1_idx]
        p2 = vertices[p2_idx]

        # Middle point
        mx = (p1[0] + p2[0]) / 2.0
        my = (p1[1] + p2[1]) / 2.0
        mz = (p1[2] + p2[2]) / 2.0

        # Normalize to sphere surface
        length = math.sqrt(mx*mx + my*my + mz*mz)
        mx /= length
        my /= length
        mz /= length

        vertices.append([mx, my, mz])
        idx = len(vertices) - 1
        vertex_cache[key] = idx
        return idx

    for _ in range(subdivisions):
        new_faces = []
        vertex_cache.clear()

        for face in faces:
            v1, v2, v3 = face

            # Get midpoints
            a = get_middle_point(v1, v2)
            b = get_middle_point(v2, v3)
            c = get_middle_point(v3, v1)

            # Create 4 new triangles
            new_faces.append([v1, a, c])
            new_faces.append([v2, b, a])
            new_faces.append([v3, c, b])
            new_faces.append([a, b, c])

        faces = new_faces

    # Scale to radius and prepare output
    vertices_arr = np.array(vertices, dtype=np.float32) * radius
    normals_arr = np.array(vertices, dtype=np.float32)  # Unit normals
    indices_arr = np.array(faces, dtype=np.uint32)

    return vertices_arr, normals_arr, indices_arr


def sphere_vertex_count(lat_segments: int, lon_segments: int) -> int:
    """
    Calculate vertex count for a lat-long sphere.

    Args:
        lat_segments: Number of latitude divisions
        lon_segments: Number of longitude divisions

    Returns:
        Total number of vertices
    """
    return (lat_segments + 1) * (lon_segments + 1)


def sphere_triangle_count(lat_segments: int, lon_segments: int) -> int:
    """
    Calculate triangle count for a lat-long sphere.

    Args:
        lat_segments: Number of latitude divisions
        lon_segments: Number of longitude divisions

    Returns:
        Total number of triangles
    """
    return lat_segments * lon_segments * 2


def icosphere_vertex_count(subdivisions: int) -> int:
    """
    Calculate approximate vertex count for icosphere.

    Args:
        subdivisions: Number of subdivision iterations

    Returns:
        Approximate vertex count
    """
    # Starts with 12, each subdivision roughly quadruples faces
    # vertices ~ 10 * 4^subdivisions + 2
    return 10 * (4 ** subdivisions) + 2


def build_semicircle(radius: float = 1.0,
                     segments: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a semi-circle (half disk) using triangle fan approach.

    The semi-circle has:
    - Curved part on top (positive Y)
    - Flat edge on bottom (Y=0)
    - Faces +Z direction

    Args:
        radius: Semi-circle radius
        segments: Number of arc segments (more = smoother curve)

    Returns:
        Tuple of (vertices, normals, indices):
        - vertices: (segments+2)x3 array of vertex positions
        - normals: (segments+2)x3 array of vertex normals (all [0,0,1])
        - indices: segments x 3 array of triangle indices
    """
    vertices = []
    normals = []
    indices = []

    # Center vertex at origin (bottom center of semi-circle)
    vertices.append([0.0, 0.0, 0.0])
    normals.append([0.0, 0.0, 1.0])

    # Arc vertices from left (-X) to right (+X), curving up (+Y)
    # Angle goes from pi (180°) to 0 (0°)
    for i in range(segments + 1):
        angle = math.pi - (i / segments) * math.pi  # pi to 0
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append([x, y, 0.0])
        normals.append([0.0, 0.0, 1.0])

    # Triangle fan indices: center (0) + arc vertices
    for i in range(segments):
        indices.append([0, i + 1, i + 2])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(indices, dtype=np.uint32)
    )


def build_cube(size: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a cube with proper normals for lighting.

    Args:
        size: Cube side length

    Returns:
        Tuple of (vertices, normals, indices):
        - vertices: 24x3 array of vertex positions (6 faces × 4 verts)
        - normals: 24x3 array of vertex normals
        - indices: 12x3 array of triangle indices (6 faces × 2 triangles)
    """
    hs = size / 2.0  # half size

    # Each face has 4 vertices with shared normals per face
    # Front face (+Z)
    vertices = [
        [-hs, -hs,  hs], [ hs, -hs,  hs], [ hs,  hs,  hs], [-hs,  hs,  hs],  # 0-3
        # Back face (-Z)
        [ hs, -hs, -hs], [-hs, -hs, -hs], [-hs,  hs, -hs], [ hs,  hs, -hs],  # 4-7
        # Top face (+Y)
        [-hs,  hs,  hs], [ hs,  hs,  hs], [ hs,  hs, -hs], [-hs,  hs, -hs],  # 8-11
        # Bottom face (-Y)
        [-hs, -hs, -hs], [ hs, -hs, -hs], [ hs, -hs,  hs], [-hs, -hs,  hs],  # 12-15
        # Right face (+X)
        [ hs, -hs,  hs], [ hs, -hs, -hs], [ hs,  hs, -hs], [ hs,  hs,  hs],  # 16-19
        # Left face (-X)
        [-hs, -hs, -hs], [-hs, -hs,  hs], [-hs,  hs,  hs], [-hs,  hs, -hs],  # 20-23
    ]

    normals = [
        # Front face
        [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
        # Back face
        [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
        # Top face
        [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
        # Bottom face
        [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
        # Right face
        [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
        # Left face
        [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
    ]

    indices = [
        # Front
        [0, 1, 2], [0, 2, 3],
        # Back
        [4, 5, 6], [4, 6, 7],
        # Top
        [8, 9, 10], [8, 10, 11],
        # Bottom
        [12, 13, 14], [12, 14, 15],
        # Right
        [16, 17, 18], [16, 18, 19],
        # Left
        [20, 21, 22], [20, 22, 23],
    ]

    return (
        np.array(vertices, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(indices, dtype=np.uint32)
    )
