import numpy as np


def rotation_matrix_to_angle(R):
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    return np.degrees(theta)


def evaluate_rotation(R_estimated, R_true):
    R_diff = np.dot(R_estimated, R_true.T)
    angle_diff = rotation_matrix_to_angle(R_diff)
    return angle_diff


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))