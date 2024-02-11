import os
import os.path

from src import solver_knapsack_makselivanov


def main(dataset):
    ALGORITHMS = {
        "Admm with 3 block": solver_knapsack_makselivanov.solverQKP_3ADMM
    }
    results = {k: {} for k in ALGORITHMS}
    problems = os.listdir(dataset)
    for problem in problems:
        print(f"Working on problem: {problem}")
        problem_path = os.path.join(dataset, problem)
        # qmdmcpy
        # qmkp = qmkpy.QMKProblem.load(problem_path, strategy='txt')
        for _name, _algorithm in ALGORITHMS.items():
            qmkp.algorithm = _algorithm
            _assignments, _profit = qmkp.solve()
            results[_name][problem] = _profit
    print(results)


if __name__ == "__main__":
    THIS_DIR = os.path.dirname(__file__)
    DATASET = os.path.join(THIS_DIR, "datasets", "qmdmckp")
    main(DATASET)