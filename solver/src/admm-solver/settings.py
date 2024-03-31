import numpy as np


class Settings:
    verbose: bool = False
    max_iter: np.uint64 = 20
    rho: np.float64 = 1.403177439024107
    alpha: np.float64 = 0.55525193250061
    beta: np.float64 = 0.03390049478067943
    gamma: np.float64 = 1.0787153140698877
    mu: np.float64 = 1.139102154353725
    eps: np.float64 = 1e-6

    def __init__(self):
        pass

    def validate(self):
        kwargs = dict(rho=self.rho,
                      alpha=self.alpha,
                      beta=self.beta,
                      gamma=self.gamma,
                      mu=self.mu,
                      eps=self.eps,
                      max_iter=self.max_iter)
        for key, value in kwargs.items():
            if value < 0:
                raise ValueError(f"{key} should be not negative")
