"""SimpleLayer module."""

from typing import Optional

import numpy as np

from experiments.weights import Weights


class SimpleLayer:
    def __init__(
            self,
            n: int,
            potential_decay: float = 0.85,
            activation_threshold: float = 0.85,
            refractory_value: float = -0.20,
    ):
        self.n = n
        self.potential_decay = potential_decay
        self.activation_threshold = activation_threshold
        self.refractory_value = refractory_value

        self.potential = np.random.random(size=n)
        self.thresholds = np.ones_like(self.potential) * activation_threshold
        self.weights = Weights(n=n)

    def activations(self) -> np.ndarray:
        return (self.potential > self.thresholds).astype(float)

    def tick(self, activations: Optional[np.ndarray] = None, external_update: Optional[np.ndarray | float] = None):
        if activations is None:
            activations = self.activations()
        if external_update is None:
            external_update = 0.0
        update = self.weights.compute(activations=activations)
        future_potential = self.potential * self.potential_decay + update + external_update
        self.potential = np.where(activations > 0, self.refractory_value, future_potential)
