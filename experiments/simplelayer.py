"""SimpleLayer module."""
from typing import Optional
import numpy as np


class SimpleLayer:
    def __init__(
            self,
            n: int,
            potential_decay: float = 0.9,
            activation_threshold: float = 0.8,
            refractory_value: float = -0.2
    ):
        self.n = n
        self.potential_decay = potential_decay
        self.activation_threshold = activation_threshold
        self.refractory_value = refractory_value

        self.potential = np.random.random(size=n)
        self.thresholds = np.ones_like(self.potential) * activation_threshold

        connections = np.random.random(size=[n, n]) * 2 - 1
        self.excitatory_connections = np.where(connections > 0, connections, 0)
        self.excitatory_connections = self.excitatory_connections/ np.linalg.norm(self.excitatory_connections, axis=1)[:, np.newaxis]
        self.inhibitory_connections = np.where(connections < 0, connections, 0)
        self.inhibitory_connections = np.abs(self.inhibitory_connections) / np.linalg.norm(self.inhibitory_connections, axis=1)[:, np.newaxis]

    def activations(self) -> np.ndarray:
        return (self.potential > self.thresholds).astype(float)

    def tick(self, activations: Optional[np.ndarray] = None):
        if activations is None:
            activations = self.activations()
        excitations = self.excitatory_connections.dot(activations)
        inhibitions = self.inhibitory_connections.dot(activations)
        computed_update = excitations - inhibitions
        computed_potential = self.potential * self.potential_decay + np.where(computed_update > 0, computed_update, 0)
        self.potential = np.where(activations > 0, self.refractory_value, computed_potential)
