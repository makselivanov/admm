import os

import clarabel
import numpy as np
import qpsolvers
from scipy.sparse import csc_matrix

from loader.src import qmdmckp

# right now it's not working for problem
def solve(profits: np.ndarray | csc_matrix,
          groups: np.ndarray | csc_matrix | None = None,
          weights: np.ndarray | csc_matrix | None = None,
          capacities: np.ndarray | None = None,
          solver="clarabel", loss: None = None, **kwargs):
    N = profits.shape[0]
    if isinstance(profits, np.ndarray):
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


def load_problem(filepath):
    this_dir = os.path.dirname(__file__)
    dataset = os.path.join(this_dir, "..", "datasets", "qmdmckp")
    problem = os.path.join(dataset, filepath)
    print(f"Working on problem: {problem}")
    problem_path = os.path.join(dataset, problem)
    qmdmckp_emulator = qmdmckp.load(problem_path)
    qmdmckp_emulator.algorithm = solve
    qmdmckp_emulator.additional["verbose"] = True
    _assignments = qmdmckp_emulator.solve()
    _profit = qmdmckp_emulator.profit(_assignments)
    print(f"QP profit: {_profit=}")


if __name__ == '__main__':
    # Make default knapsack problem for testing
    print(f"Available solutions: {qpsolvers.available_solvers}")
    # load_problem("example.txt")
    load_problem("qmdmckp_100_25_1_0_1.txt")

