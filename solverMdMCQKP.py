import clarabel
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler


# Валидирует переданные данные, возвращает (N, M, K) из описания рюкзака
def validatorMdMCQ(profits: np.ndarray,
                   groups: np.ndarray,
                   weights: np.ndarray,
                   capacity: np.ndarray,
                   rtol=1e-05, atol=1e-08):
    N = profits.shape[0]
    M = capacity.shape[0]
    K = groups.shape[0]

    if len(profits.shape) != 2:
        raise ValueError("profits is not matrix (not 2d array)")
    if len(groups.shape) != 2:
        raise ValueError("groups is not matrix (not 2d array)")
    if len(weights.shape) != 2:
        raise ValueError("weights is not matrix (not 2d array)")
    if len(capacity.shape) != 1:
        raise ValueError("capacity is not vector (not 1d array)")

    if profits.shape != (N, N):
        raise ValueError("profits is not square matrix (not (N, N) matrix)")
    if groups.shape != (K, N):
        raise ValueError("groups is not (K, N) matrix")
    if weights.shape != (M, N):
        raise ValueError("weights is not (M, N) matrix")

    def isSymMatrix(matrix):
        return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)

    def isBinaryMatrix(matrix):
        return np.array_equal(matrix, matrix.astype(bool))

    if not isSymMatrix(profits):
        raise ValueError("profits is not symmetric matrix")
    if not isBinaryMatrix(groups):
        raise ValueError("groups is not binary matrix")
    return N, M, K


def solverMdMCQKP_2ADMM(profits: np.ndarray,
                        groups: np.ndarray,
                        weights: np.ndarray,
                        capacity: np.ndarray,
                        x_0: np.ndarray = None,  # need to validate from here
                        zu_0: np.ndarray = None,
                        lambda_0: np.ndarray = None,
                        epochs: np.uint64 = 20,
                        rho: np.float64 = None,
                        alpha: np.float64 = None,
                        beta: np.float64 = None,
                        mu: np.float64 = None,
                        eps: np.float64 = None,
                        ):
    # settings
    clarabel_settings = clarabel.DefaultSettings()
    # validator raise ValueError if argument is not valid
    N, M, K = validatorMdMCQ(profits, groups, weights, capacity)
    # TODO validate constant and initial values
    pass


def solverMdMCQKP_3ADMM(profits: np.ndarray,
                        groups: np.ndarray,
                        weights: np.ndarray,
                        capacity: np.ndarray,
                        x_0: np.ndarray = None,  # need to validate from here
                        zu_0: np.ndarray = None,
                        y_0: np.ndarray = None,
                        lambda_0: np.ndarray = None,
                        epochs: np.uint64 = 20,
                        rho: np.float64 = None,
                        alpha: np.float64 = None,
                        beta: np.float64 = None,
                        gamma: np.float64 = None,
                        mu: np.float64 = None,
                        eps: np.float64 = None,
                        loss=None
                        ):
    # settings
    clarabel_settings = clarabel.DefaultSettings()
    # validator raise ValueError if argument is not valid
    N, M, K = validatorMdMCQ(profits, groups, weights, capacity)
    A_1 = np.zeros((N, N + M))
    for i in range(N):
        A_1[i, i] = -1
    qubo_sampler = SimulatedAnnealingSampler()
    # TODO validate constant and initial values
    epochs += 1
    x = np.full(epochs, x_0)
    zu = np.full(epochs, zu_0)
    y = np.full(epochs, y_0)
    lamb = np.full(epochs, lambda_0)
    metrics = np.zeros(epochs)
    # TODO calculate metrics for zero
    for curr_epoch in range(1, epochs):
        prev_epoch = curr_epoch - 1
        # Qubo block
        Q = - profits + alpha / 2 * groups.T * groups + rho / 2 * np.eye(N)
        diag = - (alpha * groups.T * np.ones(K) + rho * (A_1 * zu[prev_epoch] + y[prev_epoch]) - lamb[prev_epoch])
        for i in range(N):
            Q[i, i] += diag[i]
        qubo_sampler.sample_qubo(Q)
        # TODO solve with SimulatedAnnealingSampler?
        # Convex block
        q = lamb[prev_epoch] + rho * (x[curr_epoch] - y[prev_epoch])
        cones = clarabel.NonnegativeConeT(M)
        solver = clarabel.DefaultSolver(rho * np.eye(N), q, weights, capacity, cones, clarabel_settings)
        solution = solver.solve()
        if not solution.status:
            raise "Solution not found for convex block"
        zu[curr_epoch] = np.hstack([solution.x, solution.z])
        # Convex + quadratic block
        y[curr_epoch] = (lamb[prev_epoch] + rho * (x[curr_epoch] + A_1 * zu[curr_epoch])) / (gamma + rho)
        # update lambda
        lamb[curr_epoch] = lamb[prev_epoch] + rho * (x[curr_epoch] + A_1 * zu[curr_epoch] - y[curr_epoch])
        # calculate metrics
        metrics[curr_epoch] = - x[curr_epoch].T * profits * x[curr_epoch] + mu * loss(x[curr_epoch], zu[curr_epoch])
    pass


if __name__ == '__main__':
    # Make default knapsack problem for testing
    N, M, K = 3, 1, 0
    profits = np.array([[3, 0, 0],
                        [0, 1, 0],
                        [0, 0, 5]])
    groups = np.ndarray((K, N))
    weights = np.array([[5, 1, 8]])
    capacity = np.array([8])
    generator = np.random.default_rng(1234)
    settings = {
        "x_0": generator.random(N),
        "zu_0": generator.random(N + M),
        "y_0": generator.random(M),
        "lambda_0": generator.random(M),
        "epochs": 20,
        "rho": 1e3,
        "alpha": 1e3,
        "beta": 1e3,
        "gamma": 1e3,
        "mu": 1e3,
        "eps": 1e-6,
    }
    solverMdMCQKP_3ADMM(profits, groups, weights, capacity, **settings)
