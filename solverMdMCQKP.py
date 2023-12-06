import clarabel
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from scipy import sparse


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


def fromAdjacencyMatrixToEdgeDict(matrix: np.ndarray, eps: float):
    if len(matrix.shape) != 2:
        raise ValueError("Argument is not matrix")
    N = matrix.shape[0]
    if matrix.shape != (N, N):
        raise ValueError("Matrix is not square")
    edgeDict = {}
    for i in range(N):
        for j in range(N):
            if abs(matrix[i, j]) > eps:
                edgeDict[(i, j)] = matrix[i, j]
    return edgeDict


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


def quboSolver(Q: np.ndarray, eps: np.float64):
    qubo_sampler = SimulatedAnnealingSampler()
    edges_dict = fromAdjacencyMatrixToEdgeDict(Q, eps)
    sampled = qubo_sampler.sample_qubo(edges_dict)
    solution = sampled.first.sample
    size = max(solution.keys()) + 1
    result = np.zeros(size)
    for key, item in solution.items():
        result[key] = item
    return result


def defaultLoss(x: np.ndarray, zu: np.ndarray) -> np.float64:
    A_1 = np.zeros((N, N + M))
    for i in range(N):
        A_1[i, i] = -1
    return ((x + A_1.dot(zu)) ** 2).sum()


# TODO maybe profits, groups, weights can be sparse.csc_matrix()
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
                        loss=defaultLoss
                        ):
    # settings
    clarabel_settings = clarabel.DefaultSettings()
    clarabel_settings.verbose = False
    # validator raise ValueError if argument is not valid
    N, M, K = validatorMdMCQ(profits, groups, weights, capacity)
    A_1 = np.zeros((N, N + M))
    for i in range(N):
        A_1[i, i] = -1

    # TODO validate constant and initial values
    epochs += 1
    x = np.tile(x_0, (epochs, 1))
    zu = np.tile(zu_0, (epochs, 1))
    y = np.tile(y_0, (epochs, 1))
    lamb = np.tile(lambda_0, (epochs, 1))
    metrics = np.zeros(epochs)
    metrics[0] = - x[0].T.dot(profits.dot(x[0])) + mu * loss(x[0], zu[0])
    # TODO calculate metrics for zero
    for curr_epoch in range(1, epochs):
        prev_epoch = curr_epoch - 1
        # Qubo block
        qubo_matrix = - profits + alpha / 2 * groups.T.dot(groups) + rho / 2 * np.eye(N)
        diag = - (alpha * groups.T.dot(np.ones(K)) + rho * (A_1.dot(zu[prev_epoch]) + y[prev_epoch]) - lamb[prev_epoch])
        qubo_matrix += np.diag(diag)
        # adding penalty beta / 2 * \| Wx + u - c \|^2
        # beta / 2 * |Wx + u - c|^T |Wx + u - c|
        # beta / 2 * (x^TW^TWx + 2(u-c)^TWx)
        qubo_matrix += beta / 2 * weights.T * weights
        qubo_matrix += beta * np.diag((zu[prev_epoch][N:] - capacity).T.dot(weights))
        x[curr_epoch] = quboSolver(qubo_matrix, eps)
        # Qubo is list of edges with weights
        # Convex block
        q = lamb[prev_epoch] + rho * (x[curr_epoch] - y[prev_epoch])
        cones = [clarabel.NonnegativeConeT(M)]
        matrix = sparse.csc_matrix(rho * np.eye(N))
        weights_csc = sparse.csc_matrix(weights)
        solver = clarabel.DefaultSolver(matrix, q, weights_csc, capacity, cones, clarabel_settings)
        solution = solver.solve()
        if not solution.status:
            raise "Solution not found for convex block"
        zu[curr_epoch] = np.hstack([solution.x, solution.z])
        # Convex + quadratic block
        y[curr_epoch] = (lamb[prev_epoch] + rho * (x[curr_epoch] + A_1.dot(zu[curr_epoch]))) / (gamma + rho)
        # update lambda
        lamb[curr_epoch] = lamb[prev_epoch] + rho * (x[curr_epoch] + A_1.dot(zu[curr_epoch]) - y[curr_epoch])
        # calculate metrics
        metrics[curr_epoch] = - x[curr_epoch].T.dot(profits.dot(x[curr_epoch])) + mu * loss(x[curr_epoch], zu[curr_epoch])

    best_epoch = metrics.argmin()
    return x[best_epoch]


if __name__ == '__main__':
    # Make default knapsack problem for testing
    N, M, K = 3, 1, 0
    profits = np.array([[3, 1, 0],
                        [1, 1, 10],
                        [0, 10, 5]])
    groups = np.ndarray((K, N))
    weights = np.array([[5, 1, 8]])
    capacity = np.array([8])
    generator = np.random.default_rng(1234)
    settings = {
        "x_0": generator.random(N),
        "zu_0": generator.random(N + M),
        "y_0": generator.random(N),
        "lambda_0": generator.random(N),
        "epochs": 20,
        "rho": 1e0,
        "alpha": 1e0,
        "beta": 1e1,
        "gamma": 1e0,
        "mu": 1e0,
        "eps": 1e-6,
    }
    solution = solverMdMCQKP_3ADMM(profits, groups, weights, capacity, **settings)
    print(solution)
