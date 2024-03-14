import numpy as np
import qpsolvers


def solve(profits: np.ndarray,
          groups: np.ndarray,
          weights: np.ndarray,
          capacities: np.ndarray,
          solver="clarabel"):
    x = qpsolvers.solve_qp(
        profits, np.zeros(profits.shape[0]),
        weights, capacities,
        groups, np.ones(groups.shape[0]),
        solver=solver)
    return x


if __name__ == '__main__':
    # Make default knapsack problem for testing
    print(f"Available solutions: {qpsolvers.available_solvers}")
    N, M, K = 3, 1, 0
    profits = np.array([[3, 1, 0],
                        [1, 1, 10],
                        [0, 10, 5]])
    groups = np.ndarray((K, N))
    weights = np.array([[5, 1, 8]])
    capacity = np.array([8])
    solution = solve(profits, groups, weights, capacity)
    print(f"QP solution: {solution=}")

