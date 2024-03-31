from abc import abstractmethod

import numpy as np
import scipy

from .settings import Settings


class AdmmSolver:
    def __init__(self,
                 profit: np.ndarray,
                 groups: np.ndarray,
                 weights: np.ndarray,
                 capacities: np.ndarray,
                 settings: Settings,
                 initvals: dict):
        self.settings = settings
        self.profit = profit
        self.groups = groups
        self.weights = weights
        self.capacities = capacities
        self.initvals: dict = dict()

    @abstractmethod
    def solve(self) -> np.ndarray:
        pass

    def validate(self, rtol=1e-05, atol=1e-08) -> (int, int, int):
        profits = self.profit
        groups = self.groups
        weights = self.weights
        capacities = self.capacities

        N = profits.shape[0]
        M = capacities.shape[0]
        K = groups.shape[0]

        if len(profits.shape) != 2:
            raise ValueError(f"profits is not matrix, shape is {profits.shape}")
        if len(groups.shape) != 2:
            raise ValueError(f"groups is not matrix, shape is {groups.shape}")
        if len(weights.shape) != 2:
            raise ValueError(f"weights is not matrix, shape is {weights.shape}")
        if len(capacities.shape) != 1:
            raise ValueError(f"capacity is not vector, shape is {capacities.shape}")

        if profits.shape != (N, N):
            raise ValueError(f"profits is not square matrix, shape is {profits.shape}")
        if groups.shape != (K, N):
            raise ValueError(f"groups is not (K, N) matrix, shape is {groups.shape}, K = {K}, N = {N}")
        if weights.shape != (M, N):
            raise ValueError(f"weights is not (M, N) matrix, shape is {weights.shape}, M = {M}, N = {N}")

        def is_binary_matrix(matrix):
            return np.array_equal(matrix, matrix.astype(bool))

        if not scipy.linalg.issymmetric(profits, rtol=rtol, atol=atol):
            raise ValueError("profits is not symmetric matrix")
        if not is_binary_matrix(groups):
            raise ValueError("groups is not binary matrix")
        return N, M, K
