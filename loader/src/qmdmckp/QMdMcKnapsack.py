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
        profits = sparse.csc_matrix((n, n))
        weights = sparse.csc_matrix((m, k))
        groups = sparse.csc_matrix((k, m), bool)  # FIXME maybe double?
        linear_profit_line = list(map(float, input_file.readline().split()))
        for elem, index in enumerate(linear_profit_line):
            if abs(elem) > eps:
                profits[index, index] = elem
        for i in range(n):
            quad_profit_line = list(map(float, input_file.readline().split()))
            for elem, j in enumerate(quad_profit_line):
                if abs(elem) > eps:
                    profits[i, j] = profits[j, i] = elem

        capacities = list(map(float, input_file.readline().split()))
        for i in range(m):
            weight_line = list(map(float, input_file.readline().split()))
            for elem, j in enumerate(weight_line):
                if abs(elem) > eps:
                    weights[i, j] = elem
        for i in range(k):
            group_line = list(map(bool, input_file.readline().split()))
            for elem, j in enumerate(group_line):
                if elem:
                    groups[i, j] = elem
        return QMdMcKnapsack(name,
                             groups,
                             weights,
                             capacities,
                             **kwargs)


if __name__ == '__main__':
    pass