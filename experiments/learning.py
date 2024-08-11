"""Module with learning functions."""

import numpy as np

from experiments.weights import Weights


class HebbianLearning:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def update(
            self,
            old_activations: np.ndarray,
            new_activations: np.ndarray,
            weights: Weights) -> np.ndarray:
        self.excitatory_update(old_activations=old_activations, new_activations=new_activations, weights=weights)
        self.inhibitory_update(old_activations=old_activations, new_activations=new_activations, weights=weights)

    def excitatory_update(
            self,
            old_activations: np.ndarray,
            new_activations: np.ndarray,
            weights: Weights) -> np.ndarray:
        # old_activations: n-dimensional
        # new_activations: n-dimensional
        # excitatory_connections: n x n
        matches = new_activations[:, np.newaxis].dot(old_activations[np.newaxis, :])
        # matches: n x n
        weights.excitatory_connections += matches * self.learning_rate

    def inhibitory_update(
            self,
            old_activations: np.ndarray,
            new_activations: np.ndarray,
            weights: Weights) -> np.ndarray:
        # old_activations: n-dimensional
        # new_activations: n-dimensional
        # excitatory_connections: n x n
        matches = old_activations[:, np.newaxis].dot(new_activations[np.newaxis, :])
        # matches: n x n
        weights.inhibitory_connections += matches * self.learning_rate
