from ar_filter.primitives import create_quad, create_sphere


def test_create_sphere_generates_vertices():
    vertices = create_sphere(radius=1.0, segments=6, rings=6)
    assert len(vertices) == (6 + 1) * (6 + 1)
    for x, y, z in vertices:
        assert -1.0 <= x <= 1.0
        assert -1.0 <= y <= 1.0
        assert -1.0 <= z <= 1.0


def test_create_quad_returns_four_vertices():
    quad = create_quad(center=(0.0, 0.0, 0.0), size=(2.0, 2.0))
    assert len(quad) == 4
    assert quad[0] == (-1.0, -1.0, 0.0)
    assert quad[2] == (1.0, 1.0, 0.0)
