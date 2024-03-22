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

    def profit(self, assignment):
        profit = assignment.T.dot(self.profits.dot(assignment))
        if (np.vectorize(lambda value: value > 0)(self.weights.dot(assignment) - self.capacity)).any():
            profit = 0
        return profit


def load(problem_path: str, **kwargs):
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


def metrics(problems_path, results_path, metrics_path):
    results = os.listdir(results_path)
    if not os.path.exists(metrics_path):
        os.mkdir(metrics_path)
    for result in results:
        result_path = os.path.join(results_path, result)
        with open(result_path) as result_file:
            metric_path = os.path.join(metrics_path, result)
            with open(metric_path, "w") as metric_file:
                for line in result_file:
                    name, *numbers = line.split()
                    assignment = np.array(list(map(int, numbers)))
                    problem = os.path.join(problems_path, name)
                    knapsack = load(problem)
                    metric = knapsack.profit(assignment)
                    metric_file.write(f"{name} {metric}\n")
