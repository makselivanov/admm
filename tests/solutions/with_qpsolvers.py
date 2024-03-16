import clarabel
import numpy as np
import qpsolvers
from scipy.sparse import csc_matrix


def solve(profits: np.ndarray | csc_matrix,
          groups: np.ndarray | csc_matrix,
          weights: np.ndarray | csc_matrix,
          capacities: np.ndarray,
          solver="clarabel", **kwargs):
    x = None
    if groups.shape[0] == 0:
        x = qpsolvers.solve_qp(
            profits, np.zeros(profits.shape[0]),
            weights, capacities,
            groups, np.ones(groups.shape[0]),
            solver=solver, **kwargs)
    else:
        x = qpsolvers.solve_qp(
            profits, np.zeros(profits.shape[0]),
            weights, capacities,
            groups, np.ones(groups.shape[0]),
            solver=solver, **kwargs)
    return x


if __name__ == '__main__':
    # Make default knapsack problem for testing
    print(f"Available solutions: {qpsolvers.available_solvers}")

    M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
    profits = np.dot(M.T, M)  # quick way to build a symmetric matrix
    # q = dot(array([3., 2., 3.]), M).reshape((3,))
    weights = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
    capacity = np.array([3., 2., -2.]).reshape((3,))
    groups = np.array([[1., 1., 1.]])
    b = np.array([1.])

    # N, M, K = 3, 1, 0
    # profits = csc_matrix([[3, 1, 0],
    #                       [1, 1, 10],
    #                       [0, 10, 5]])
    # groups = csc_matrix((K, N))
    # weights = csc_matrix([[5, 1, 8]])
    # capacity = np.array([8])
    solution = solve(profits, groups, weights, capacity, time_limit=100)
    print(f"QP solution: {solution=}")

