import clarabel
import numpy as np
import qpsolvers
from scipy.sparse import csc_matrix


def solve(profits: np.ndarray | csc_matrix,
          groups: np.ndarray | csc_matrix | None,
          weights: np.ndarray | csc_matrix | None,
          capacities: np.ndarray | None,
          solver="clarabel", **kwargs):
    N = profits.shape[0]
    if isinstance(weights, np.ndarray):
        profits = csc_matrix(profits)
    kwargs["P"] = -profits
    kwargs["q"] = np.zeros(N)
    if isinstance(weights, np.ndarray):
        weights = csc_matrix(weights)
    kwargs["G"] = weights
    kwargs["h"] = capacities
    if isinstance(groups, np.ndarray):
        groups = csc_matrix(groups)
    if groups is not None and groups.shape[0] != 0:
        kwargs["A"] = groups
        kwargs["b"] = np.ones(groups.shape[0])
    kwargs["lb"] = np.zeros(N)
    kwargs["ub"] = np.ones(N)
    kwargs["solver"] = solver
    x = qpsolvers.solve_qp(**kwargs)
    if isinstance(x, np.ndarray):
        x = np.digitize(x, [0.5])
    return x


if __name__ == '__main__':
    # Make default knapsack problem for testing
    print(f"Available solutions: {qpsolvers.available_solvers}")

    M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    profits = np.dot(M.T, M)  # quick way to build a symmetric matrix
    # q = dot(array([3., 2., 3.]), M).reshape((3,))
    weights = np.array([[1., 2., 1.], [2., 0., 1.], [1., 2., 1.]])
    capacity = np.array([3., 2., 2.]).reshape((3,))
    groups = None

    # N, M, K = 3, 1, 0
    # profits = csc_matrix([[3, 1, 0],
    #                       [1, 1, 10],
    #                       [0, 10, 5]])
    # groups = csc_matrix((K, N))
    # weights = csc_matrix([[5, 1, 8]])
    # capacity = np.array([8])
    solution = solve(profits, groups, weights, capacity, verbose=True)
    print(f"QP solution: {solution=}")

