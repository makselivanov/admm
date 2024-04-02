import numpy as np
from abc import abstractmethod


class Loss:
    @abstractmethod
    def __call__(self, x: np.ndarray, zu: np.ndarray) -> np.float64:
        raise NotImplementedError
