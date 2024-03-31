import clarabel
import numpy as np
import scipy.linalg
from dwave.samplers import SimulatedAnnealingSampler
from scipy import sparse


# Валидирует переданные данные, возвращает (N, M, K) из описания рюкзака
def validateMdMCQ(profits: np.ndarray,
                  groups: np.ndarray,
                  weights: np.ndarray,
                  capacity: np.ndarray,
                  rtol=1e-05, atol=1e-08):
    N = profits.shape[0]
    M = capacity.shape[0]
    K = groups.shape[0]

    if len(profits.shape) != 2:
        raise ValueError(f"profits is not matrix, shape is {profits.shape}")
    if len(groups.shape) != 2:
        raise ValueError(f"groups is not matrix, shape is {groups.shape}")
    if len(weights.shape) != 2:
        raise ValueError(f"weights is not matrix, shape is {weights.shape}")
    if len(capacity.shape) != 1:
        raise ValueError(f"capacity is not vector, shape is {capacity.shape}")

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


def fromAdjacencyMatrixToEdgeDict(matrix: np.ndarray, eps: float):
    if len(matrix.shape) != 2:
        raise ValueError("Argument is not matrix")
    N = matrix.shape[0]
    if matrix.shape != (N, N):
        raise ValueError("Matrix is not square")
    edge_dict = {}
    for i in range(N):
        for j in range(N):
            if abs(matrix[i, j]) > eps:
                edge_dict[(i, j)] = matrix[i, j]
    return edge_dict


# def solverMdMCQKP_2ADMM(profits: np.ndarray,
#                         groups: np.ndarray,
#                         weights: np.ndarray,
#                         capacity: np.ndarray,
#                         x_0: np.ndarray = None,  # need to validate from here
#                         zu_0: np.ndarray = None,
#                         lambda_0: np.ndarray = None,
#                         epochs: np.uint64 = 20,
#                         rho: np.float64 = None,
#                         alpha: np.float64 = None,
#                         beta: np.float64 = None,
#                         mu: np.float64 = None,
#                         eps: np.float64 = None,
#                         ):
#     # settings
#     clarabel_settings = clarabel.DefaultSettings()
#     # validator raise ValueError if argument is not valid
#     N, M, K = validateMdMCQ(profits, groups, weights, capacity)
#     # TODO validate constant and initial values
#     pass


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


def validateConstants(**kwargs):
    for key, value in kwargs.items():
        if value < 0:
            raise ValueError(f"{key} should be not negative")


def validateInitialValues(x_0, zu_0, y_0, lambda_0, N, M):
    if x_0.shape != (N, ):
        raise ValueError("Initial 'x_0' shape should be equal to amount of items")
    if y_0.shape != (N, ):
        raise ValueError("Initial 'y_0' shape should be equal to amount of items")
    if lambda_0.shape != (N, ):
        raise ValueError("Initial 'lambda_0' shape should be equal to amount of items")
    if zu_0.shape != (N + M, ):
        raise ValueError("Initial 'zu_0' shape should be equal to amount of items plus amount of objectives dimensions")


def defaultLoss(x: np.ndarray, zu: np.ndarray) -> np.float64:
    N = x.shape[0]
    NM = zu.shape[0]
    A_1 = np.zeros((N, NM))
    for i in range(N):
        A_1[i, i] = -1
    return ((x + A_1.dot(zu)) ** 2).sum()


def solverMdMCQKP_3ADMM(profits: np.ndarray,
                        groups: np.ndarray,
                        weights: np.ndarray,
                        capacities: np.ndarray,
                        x_0: np.ndarray = None,  # need to validate from here
                        zu_0: np.ndarray = None,
                        y_0: np.ndarray = None,
                        lambda_0: np.ndarray = None,
                        epochs: np.uint64 = 20,
                        rho: np.float64 = 1.403177439024107,
                        alpha: np.float64 = 0.55525193250061,
                        beta: np.float64 = 0.03390049478067943,
                        gamma: np.float64 = 1.0787153140698877,
                        mu: np.float64 = 1.139102154353725,
                        eps: np.float64 = 1e-6,
                        loss=defaultLoss
                        ):
    # settings
    clarabel_settings = clarabel.DefaultSettings()
    clarabel_settings.verbose = False
    # validator raise ValueError if argument is not valid
    N, M, K = validateMdMCQ(profits, groups, weights, capacities)
    A_1 = np.zeros((N, N + M))
    for i in range(N):
        A_1[i, i] = -1

    validateConstants(epochs=epochs, rho=rho, alpha=alpha, beta=beta, gamma=gamma, mu=mu, eps=eps)
    if x_0 is None:
        x_0 = np.random.random_integers(0, 1, N)
    if zu_0 is None:
        zu_0 = np.random.rand(N + M)
    if y_0 is None:
        y_0 = np.random.rand(N)
    if lambda_0 is None:
        lambda_0 = np.random.rand(N)
    validateInitialValues(x_0, zu_0, y_0, lambda_0, N, M)
    epochs += 1
    x = np.tile(x_0, (epochs, 1))
    zu = np.tile(zu_0, (epochs, 1))
    y = np.tile(y_0, (epochs, 1))
    lamb = np.tile(lambda_0, (epochs, 1))
    metrics = np.zeros(epochs)
    metrics[0] = - x[0].T.dot(profits.dot(x[0])) + mu * loss(x[0], zu[0])
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
        qubo_matrix += beta * np.diag((zu[prev_epoch][N:] - capacities).T.dot(weights))
        x[curr_epoch] = quboSolver(qubo_matrix, eps)
        # Qubo is list of edges with weights
        # Convex block
        q = lamb[prev_epoch] + rho * (x[curr_epoch] - y[prev_epoch])
        cones = [clarabel.NonnegativeConeT(M)]
        matrix = sparse.csc_matrix(rho * np.eye(N))
        weights_csc = sparse.csc_matrix(weights)
        solver = clarabel.DefaultSolver(matrix, q, weights_csc, capacities, cones, clarabel_settings)
        solution = solver.solve()
        if not solution.status:
            raise "Solution not found for convex block"
        zu[curr_epoch] = np.hstack([solution.x, solution.z])
        # Convex + quadratic block
        y[curr_epoch] = (lamb[prev_epoch] + rho * (x[curr_epoch] + A_1.dot(zu[curr_epoch]))) / (gamma + rho)
        # update lambda
        lamb[curr_epoch] = lamb[prev_epoch] + rho * (x[curr_epoch] + A_1.dot(zu[curr_epoch]) - y[curr_epoch])
        # calculate metrics
        metrics[curr_epoch] = - x[curr_epoch].T.dot(profits.dot(x[curr_epoch])) + mu * loss(x[curr_epoch],
                                                                                            zu[curr_epoch])

    best_epoch = metrics.argmin()
    return x[best_epoch]


def solverMdQKP_3ADMM(profits: np.ndarray,
                      weights: np.ndarray,
                      capacity: np.ndarray,
                      **kwargs):
    N = profits.shape[0]
    groups = np.ndarray((0, N))
    solverMdMCQKP_3ADMM(profits=profits, groups=groups, weights=weights, capacities=capacity, **kwargs)


def solverQKP_3ADMM(profits: np.ndarray,
                      weights: np.ndarray,
                      capacity: np.ndarray,
                      **kwargs):
    weights = weights.reshape((1, -1))
    solverMdQKP_3ADMM(profits=profits, weights=weights, capacity=capacity, **kwargs)


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
    }
    solution = solverMdMCQKP_3ADMM(profits, groups, weights, capacity, **settings)
    print(solution)
