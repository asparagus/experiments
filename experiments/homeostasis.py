"""Module with homeostasis functions."""

import numpy as np


def normalize_inputs(connections: np.ndarray) -> np.ndarray:
    return normalize(connections, new_axis=1)


def normalize_outputs(connections: np.ndarray) -> np.ndarray:
    return normalize(connections, new_axis=0)


def normalize(connections: np.ndarray, new_axis: int) -> np.ndarray:
    norms = np.expand_dims(np.linalg.norm(connections, axis=new_axis), axis=new_axis)
    return connections / norms
