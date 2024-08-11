"""Input / Output module."""

import numpy as np


class CyclicEncoder:
    def __init__(self, dim_in: int, dim_out: int, width: int):
        self.dim_in = dim_in
        self.dim_out = dim_out
        assert dim_out > width
        self.sample = np.concatenate([np.ones(width), np.zeros(dim_out - width)])
        self.overlap = dim_out / dim_in

    def encode(self, value: int) -> np.ndarray:
        return np.roll(self.sample, shift=int(self.overlap * (value % self.dim_in)))
