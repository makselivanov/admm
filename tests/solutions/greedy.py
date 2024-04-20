import os

import numpy as np
from scipy.sparse import csc_matrix

from loader.src import qmdmckp


def _is_valid_assignments(assignments, groups, weights, capacities):
    return np.allclose(groups @ assignments, np.ones(groups.shape[0])) and np.all(weights @ assignments < capacities)


def solve(profits: np.ndarray | csc_matrix,
          groups: np.ndarray | csc_matrix | None = None,
          weights: np.ndarray | csc_matrix | None = None,
          capacities: np.ndarray | None = None,
          loss: None = None,
          data_dir_path: str | None = None,
          problem_name: str | None = None):
    n = profits.shape[0]
    _assignments = np.zeros(n, dtype=np.int8)
    for i in range(n):
        _assignments[i] = 1
        if not _is_valid_assignments(_assignments, groups, weights, capacities):
            _assignments[i] = 0
    return _assignments


def load_problem(filepath):
    this_dir = os.path.dirname(__file__)
    dataset = os.path.join(this_dir, "..", "datasets", "qmdmckp")
    problem = os.path.join(dataset, filepath)
    print(f"Working on problem: {problem}")
    problem_path = os.path.join(dataset, problem)
    qmdmckp_emulator = qmdmckp.load(problem_path)
    qmdmckp_emulator.algorithm = solve
    _assignments = qmdmckp_emulator.solve()
    _profit = qmdmckp_emulator.profit(_assignments)
    print(f"QP profit: {_profit=}")


if __name__ == '__main__':
    load_problem("example.txt")
    # load_problem("qmdmckp_100_25_1_0_1.txt")

