import os

import clarabel
import numpy as np
import qpsolvers
from scipy.sparse import csc_matrix

from loader.src import qmdmckp


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
    if x is None:
        x = np.zeros(N, dtype=int)
    return x


if __name__ == '__main__':
    # Make default knapsack problem for testing
    print(f"Available solutions: {qpsolvers.available_solvers}")
    this_dir = os.path.dirname(__file__)
    dataset = os.path.join(this_dir, "..", "datasets", "qmdmckp")
    problems = os.listdir(dataset)
    problem = problems[0]
    print(f"Working on problem: {problem}")
    problem_path = os.path.join(dataset, problem)
    qmdmckp_emulator = qmdmckp.load(problem_path)
    qmdmckp_emulator.algorithm = solve
    qmdmckp_emulator.additional["verbose"] = True
    _assignments = qmdmckp_emulator.solve()
    _profit = qmdmckp_emulator.profit(_assignments)
    print(f"QP profit: {_profit=}")

