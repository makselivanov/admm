from abc import abstractmethod

import numpy as np
import scipy

from .settings import Settings
from .loss import Loss
from .losses.default_loss import DefaultLoss


class AdmmSolver:
    def __init__(self,
                 profits: np.ndarray,
                 groups: np.ndarray,
                 weights: np.ndarray,
                 capacities: np.ndarray,
                 settings: Settings = Settings(),
                 initvals: dict | None = None,
                 loss: Loss = DefaultLoss()):
        """
        Initialize solver
        :param profits: Real matrix NxN, indicate profit for each item and its combinations of two. Total profit is x^T P x
        :param groups: Binary matrix KxN, from each group will be taken only one item
        :param weights: Real matrix MxN, indicate weight for each item
        :param capacities: Real vector M, indicate capacity for each dimension
        :param settings: Settings object
        :param initvals: Initial values, key can be x, y, zu, lambda
        :param loss: Loss object
        """
        if initvals is None:
            initvals = dict()
        self.settings = settings
        self.profits = profits
        self.groups = groups
        self.weights = weights
        self.capacities = capacities
        self.initvals: dict = initvals
        self.loss = loss

        # Number of items
        self.N = profits.shape[0]
        # Number of dimensions
        self.M = capacities.shape[0]
        # Number of groups
        self.K = groups.shape[0]

    @abstractmethod
    def solve(self) -> np.ndarray:
        """
        Abstract method to solve quadratic multidimensional multi-choice knapsack problem (QMdMCKP)
        """
        raise NotImplementedError()

    def validate(self, rtol=1e-05, atol=1e-08):
        """
        Validate all parameters for QMdMCKP
        :param rtol: Relative error bound
        :type rtol: float | None
        :param atol: Absolute error bound
        :type atol: float | None
        :raise ValueError: If parameter is invalid
        """
        self._validate_dimensions()
        self._validate_shape()
        self._validate_group_is_binary()
        self._validate_profits_is_symmetric(rtol, atol)

    def _validate_shape(self):
        if self.profits.shape != (self.N, self.N):
            raise ValueError(f"profits is not square matrix, shape is {self.profits.shape}")
        if self.groups.shape != (self.K, self.N):
            raise ValueError(f"groups is not (K, N) matrix, shape is {self.groups.shape}, K = {self.K}, N = {self.N}")
        if self.weights.shape != (self.M, self.N):
            raise ValueError(f"weights is not (M, N) matrix, shape is {self.weights.shape}, M = {self.M}, N = {self.N}")
        if self.capacities.shape != (self.M,):
            raise ValueError(f"capacities is not (M, ) vector, shape is {self.weights.shape}, M = {self.M}")

    def _validate_dimensions(self):
        if len(self.profits.shape) != 2:
            raise ValueError(f"profits is not matrix, shape is {self.profits.shape}")
        if len(self.groups.shape) != 2:
            raise ValueError(f"groups is not matrix, shape is {self.groups.shape}")
        if len(self.weights.shape) != 2:
            raise ValueError(f"weights is not matrix, shape is {self.weights.shape}")
        if len(self.capacities.shape) != 1:
            raise ValueError(f"capacity is not vector, shape is {self.capacities.shape}")

    def _validate_group_is_binary(self):
        if not np.array_equal(self.groups, self.groups.astype(bool)):
            raise ValueError("groups is not binary matrix")

    def _validate_profits_is_symmetric(self, rtol, atol):
        if not scipy.linalg.issymmetric(self.profits, rtol=rtol, atol=atol):
            raise ValueError("profits is not symmetric matrix")