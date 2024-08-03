"""Module with learning functions."""

import numpy as np


class HebbianLearning:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def excitatory_update(
            self,
            old_activations: np.ndarray,
            new_activations: np.ndarray,
            excitatory_connections: np.ndarray) -> np.ndarray:
        # old_activations: n-dimensional
        # new_activations: n-dimensional
        # excitatory_connections: n x n
        matches = new_activations[:, np.newaxis].dot(old_activations[np.newaxis, :])
        # matches: n x n
        return excitatory_connections + matches * self.learning_rate

    def inhibitory_update(
            self,
            old_activations: np.ndarray,
            new_activations: np.ndarray,
            inhibitory_connections: np.ndarray) -> np.ndarray:
        # old_activations: n-dimensional
        # new_activations: n-dimensional
        # excitatory_connections: n x n
        matches = old_activations[:, np.newaxis].dot(new_activations[np.newaxis, :])
        # matches: n x n
        return inhibitory_connections + matches * self.learning_rate
