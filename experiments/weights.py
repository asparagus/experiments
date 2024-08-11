"""Weights module."""

import numpy as np


class Weights:
    def __init__(self, n: int):
        connections = np.random.random(size=[n, n]) * 2 - 1
        self.excitatory_connections = normalize_outputs(np.where(connections > 0, connections, 0))
        self.inhibitory_connections = normalize_outputs(np.where(connections < 0, np.abs(connections), 0))

    def compute(self, activations: np.ndarray) -> np.ndarray:
        excitations = self.excitatory_connections.dot(activations)
        inhibitions = self.inhibitory_connections.dot(activations)
        diff = excitations - inhibitions
        return np.where(diff > 0, diff, 0)

    def compute_transposed(self, activations: np.ndarray) -> np.ndarray:
        excitations = activations.dot(self.excitatory_connections)
        inhibitions = activations.dot(self.inhibitory_connections)
        diff = excitations - inhibitions
        return np.where(diff > 0, diff, 0).T

    def normalize_inputs(self):
        self.excitatory_connections = normalize_inputs(self.excitatory_connections)
        self.inhibitory_connections = normalize_inputs(self.inhibitory_connections)

    def normalize_outputs(self):
        self.excitatory_connections = normalize_outputs(self.excitatory_connections)
        self.inhibitory_connections = normalize_outputs(self.inhibitory_connections)


def normalize_inputs(connections: np.ndarray) -> np.ndarray:
    return normalize(connections, new_axis=1)


def normalize_outputs(connections: np.ndarray) -> np.ndarray:
    return normalize(connections, new_axis=0)


def normalize(connections: np.ndarray, new_axis: int) -> np.ndarray:
    norms = np.expand_dims(np.linalg.norm(connections, axis=new_axis), axis=new_axis)
    return connections / norms
