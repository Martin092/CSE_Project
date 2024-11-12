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