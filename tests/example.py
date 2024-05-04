import os
import os.path
from datetime import datetime

from solver.src import admm_solver
from loader.src import qmdmckp
import solutions.with_qpsolvers as qpsolvers
import solutions.greedy as greedy
from solver.src.admm_solver import Settings


def main(dataset):
    ALGORITHMS = {
        "Admm with 3 block": admm_solver.AdmmBlock3Solver,
        "QPsolvers": qpsolvers.solve,
        "Greedy": greedy.solve,
    }
    profits = {k: {} for k in ALGORITHMS}
    assignments = {k: {} for k in ALGORITHMS}
    problems = os.listdir(dataset)
    start = datetime.now()
    for index, problem in enumerate(problems):
        print(f"Working on problem {index+1}/{len(problems)}: {problem}")
        problem_path = os.path.join(dataset, problem)
        # qmdmckp
        qmdmckp_emulator = qmdmckp.load(problem_path)

        qmdmckp_emulator.additional["loss"] = lambda x, zu: qmdmckp_emulator.loss(x)
        qmdmckp_emulator.additional["data_dir_path"] = "data"
        qmdmckp_emulator.additional["problem_name"] = problem
        for _name, _algorithm in ALGORITHMS.items():
            qmdmckp_emulator.algorithm = _algorithm
            _assignments = qmdmckp_emulator.solve()
            _profit = qmdmckp_emulator.profit(_assignments)
            profits[_name][problem] = _profit
            assignments[_name][problem] = _assignments
    finish = datetime.now()
    time_duration = finish - start
    print(profits)
    print(f"Duration of calculation: {time_duration}")
    root_dataset = os.path.split(os.path.split(dataset)[0])[0]
    result_set = os.path.join(root_dataset, "results")
    metric_set = os.path.join(root_dataset, "profits")
    qmdmckp.save(result_set, assignments)
    qmdmckp.metrics(dataset, result_set, metric_set)


if __name__ == "__main__":
    THIS_DIR = os.path.dirname(__file__)
    DATASET = os.path.join(THIS_DIR, "datasets", "qmdmckp")
    main(DATASET)
