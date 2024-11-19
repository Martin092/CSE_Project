import numpy as np

def rotation_x(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)

    return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])

def rotation_y(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)

    return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])

def rotation_z(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)

    return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

def rotation_matrix(normal, angle):
    normal = normal / np.linalg.norm(normal)
    x, y, z = normal
    cos = np.cos(angle)
    sin = np.sin(angle)
    one_minus_cos = 1 - cos

    return np.array([
        [cos + x*x*one_minus_cos, x*y*one_minus_cos - z*sin, x*z*one_minus_cos + y*sin],
        [y*x*one_minus_cos + z*sin, cos + y*y*one_minus_cos, y*z*one_minus_cos - x*sin],
        [z*x*one_minus_cos - y*sin, z*y*one_minus_cos + x*sin, cos + z*z*one_minus_cos]
    ])