import numpy as np


class Settings:
    verbose: bool = False
    max_iter: np.uint64 = 100
    rho: np.float64 = 1e5 # make 1e4 and change how its change by 0.1 until 1e7
    alpha: np.float64 = 1e5
    beta: np.float64 = 1e5
    gamma: np.float64 = 1e4 # make 1e3
    mu: np.float64 = 1e5
    eps: np.float64 = 1e-6

    def validate(self):
        for key, value in self.__dict__:
            if value < 0:
                raise ValueError(f"{key} should be not negative")
