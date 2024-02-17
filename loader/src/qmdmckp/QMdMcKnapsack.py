import os.path

import numpy as np
from scipy import sparse


class QMdMcKnapsack:
    def __init__(self, name, profits, groups, weights, capacity, **kwargs):
        self.algorithm = None
        self.name = name
        self.profits = profits
        self.groups = groups
        self.weights = weights
        self.capacity = capacity
        self.additional = kwargs

    # Return tuple of assignments and profits
    def solve(self) -> (np.ndarray, float):
        return self.algorithm(self.profits,
                              self.groups,
                              self.weights,
                              self.capacity,
                              **self.additional)


def load(problem_path: str, eps: float = 1e-6, **kwargs):
    with open(problem_path) as input_file:
        name = input_file.readline()
        n = int(input_file.readline())
        m = int(input_file.readline())
        k = int(input_file.readline())
        profits = np.zeros((n, n))
        weights = np.zeros((m, n))
        groups = np.zeros((k, n), dtype=np.int8)
        linear_profit_line = list(map(float, input_file.readline().split()))
        for index, elem in enumerate(linear_profit_line):
            profits[index, index] = elem
        for i in range(n):
            quad_profit_line = list(map(float, input_file.readline().split()))
            for j, elem in enumerate(quad_profit_line):
                profits[i, j] = profits[j, i] = elem

        capacities = np.array(list(map(float, input_file.readline().split())))
        for i in range(m):
            weight_line = list(map(float, input_file.readline().split()))
            for j, elem in enumerate(weight_line):
                weights[i, j] = elem
        for i in range(k):
            group_line = list(map(bool, input_file.readline().split()))
            for j, elem in enumerate(group_line):
                groups[i, j] = elem
        return QMdMcKnapsack(name,
                             profits,
                             groups,
                             weights,
                             capacities,
                             **kwargs)


def save(result_path: str, assignments: dict):
    for algorithm, assign_by_algorithm in assignments.items():
        path = os.path.join(result_path, algorithm + ".txt")
        with open(path, "w") as result_file:
            for problem_name, assign in assign_by_algorithm.items():
                buffer = " ".join(list(map(str, assign)))
                result_file.write(f"{problem_name} {buffer}\n")
