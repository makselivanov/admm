import os
import os.path

from solver.src import solver_knapsack_makselivanov
from loader.src import qmdmckp

def main(dataset):
    ALGORITHMS = {
        "Admm with 3 block": solver_knapsack_makselivanov.solverMdMCQKP_3ADMM
    }
    results = {k: {} for k in ALGORITHMS}
    problems = os.listdir(dataset)
    for problem in problems:
        print(f"Working on problem: {problem}")
        problem_path = os.path.join(dataset, problem)
        # qmdmcpkp
        qmdmckp_emulator = qmdmckp.load(problem_path)
        for _name, _algorithm in ALGORITHMS.items():
            qmdmckp_emulator.algorithm = _algorithm
            _assignments, _profit = qmdmckp_emulator.solve()
            results[_name][problem] = _profit
    print(results)


if __name__ == "__main__":
    THIS_DIR = os.path.dirname(__file__)
    DATASET = os.path.join(THIS_DIR, "datasets", "qmdmckp")
    main(DATASET)