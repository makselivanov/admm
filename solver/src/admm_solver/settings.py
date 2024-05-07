import numpy as np


class Settings:
    verbose: bool = False
    max_iter: np.uint64 = 100
    rho: np.float64 = 1e4  # change how its change by 0.1 until 1e7
    alpha: np.float64 = 1e6
    beta: np.float64 = 1e6
    gamma: np.float64 = 1e4
    mu: np.float64 = 1e6
    eps: np.float64 = 1e-6

    def validate(self):
        for key, value in self.__dict__:
            if value < 0:
                raise ValueError(f"{key} should be not negative")
