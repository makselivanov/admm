import datetime
import os
import time

import numpy as np
from scipy import optimize

from solver.src import solver_knapsack_makselivanov
from loader.src import qmdmckp

VAR_SIZE = 5
DATASET = "datasets/qmdmckp"
loaded_problems = []
INF = 1e9


def wrapper(x):
    kwargs = {
        "rho": x[0],
        "alpha": x[1],
        "beta": x[2],
        "gamma": x[3],
        "mu": x[4],
    }
    metrics = []
    for problem in loaded_problems:
        problem.additional = kwargs
        problem.algorithm = solver_knapsack_makselivanov.solverMdMCQKP_3ADMM
        try:
            assignments, profit = problem.solve()
            metric = problem.profit(assignments)
        except ValueError:
            metric = -INF
        metrics.append(metric / problem.profits.trace())  # Maybe sum or trace
    return -sum(metrics)


def main():
    low = [1e-3] * VAR_SIZE
    up = [1e3] * VAR_SIZE
    initial = [1e0] * VAR_SIZE
    #bounds = optimize.Bounds(low, up, keep_feasible=[True] * VAR_SIZE)

    problems = os.listdir(DATASET)
    for problem in problems:
        if problem.startswith("qmdmckp_100"):
            problem_path = os.path.join(DATASET, problem)
            qmdmckp_emulator = qmdmckp.load(problem_path)
            loaded_problems.append(qmdmckp_emulator)

    result = optimize.basinhopping(wrapper, initial)
    print(result)
    with open("optimization_annealing_results.txt", "w") as output:
        kwargs = {
            "rho": result.x[0],
            "alpha": result.x[1],
            "beta": result.x[2],
            "gamma": result.x[3],
            "mu": result.x[4],
        }
        output.write(f"Result\n{result.x}\n")
        output.write(f"Result in kwargs\n{kwargs}\n")
        output.write(f"Cause of termination: {result.message}\n")
        global_time = datetime.datetime.now()
        output.write(f"ISO Timestamp: {global_time.isoformat()}")


if __name__ == "__main__":
    main()


