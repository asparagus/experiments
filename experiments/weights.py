"""Weights module."""

import numpy as np

from experiments.homeostasis import normalize_outputs


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
