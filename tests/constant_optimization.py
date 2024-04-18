from datetime import datetime
import os

import numpy as np
from scipy import optimize

from solver.src import admm_solver
from loader.src import qmdmckp
from solver.src.admm_solver import Settings

VAR_SIZE = 5
DATASET = "datasets/qmdmckp"
loaded_problems = []
INF = 1e9


def wrapper(x):
    current_settings = Settings()
    current_settings.rho = x[0]
    current_settings.alpha = x[1]
    current_settings.beta = x[2]
    current_settings.gamma = x[3]
    current_settings.mu = x[4]
    metrics = []
    for problem in loaded_problems:
        problem.settings = current_settings
        problem.algorithm = admm_solver.AdmmBlock3Solver
        try:
            assignments, profit = problem.solve()
            metric = problem.profits(assignments)
            if metric == 0:
                metric = -INF
        except ValueError:
            metric = -INF
        metrics.append(metric / problem.profits.trace())  # Maybe sum or trace  
    return -sum(metrics)


def main():
    initial = [
        Settings.rho,
        Settings.alpha,
        Settings.beta,
        Settings.gamma,
        Settings.mu,
    ]

    problems = os.listdir(DATASET)
    for problem in problems:
        if problem.startswith("qmdmckp_100"):
            problem_path = os.path.join(DATASET, problem)
            qmdmckp_emulator = qmdmckp.load(problem_path)
            loaded_problems.append(qmdmckp_emulator)

    start = datetime.now()
    result = optimize.basinhopping(wrapper, initial, disp=True)  # TODO add callback?
    finish = datetime.now()
    time_duration = finish - start
    print(result)
    with open("optimization_results.txt", "w") as output:
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
        global_time = datetime.now()
        output.write(f"Duration of calculation: {time_duration}")
        output.write(f"ISO Timestamp: {global_time.isoformat()}")


if __name__ == "__main__":
    main()
