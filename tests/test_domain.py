import numpy as np

from finite_element_networks.domain import inside_angle, project_onto


def test_inside_angle():
    a = np.array([-1.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 1.0])
    assert np.isclose(inside_angle(a, b, c), np.pi / 4)


def test_projects_point_onto_line():
    a = np.array([0.0, -1.0])
    b = np.array([1.0, 1.0])
    p = np.array([-1.0, 2.0])
    assert np.all(np.isclose(project_onto(p, [a, b]), np.array([1.0, 1.0])))
