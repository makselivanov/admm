import json
import os
from abc import ABC

import clarabel
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler
from scipy.sparse import csc_matrix

from ..solver import AdmmSolver
from ..settings import Settings
from ..loss import Loss
from ..losses.default_loss import DefaultLoss


class AdmmBlock3Solver(AdmmSolver, ABC):
    def __init__(self,
                 profits: np.ndarray,
                 groups: np.ndarray,
                 weights: np.ndarray,
                 capacities: np.ndarray,
                 settings: Settings = Settings(),
                 initvals: dict | None = None,
                 loss: Loss = DefaultLoss(),
                 data_dir_path: str | None = None,
                 problem_name: str | None = None):
        super().__init__(profits, groups, weights, capacities, settings, initvals, loss)
        self.clarabel_settings = clarabel.DefaultSettings()
        self.clarabel_settings.verbose = False

        x_0_default = np.random.random_integers(0, 1, self.N)
        self.x_0 = self.initvals.get("x", x_0_default)

        y_0_default = np.random.rand(self.N)
        self.y_0 = self.initvals.get("y", y_0_default)

        zu_0_default = np.random.rand(self.N + self.M)
        self.zu_0 = self.initvals.get("zu", zu_0_default)

        lambda_0_default = np.random.rand(self.N)
        self.lambda_0 = self.initvals.get("lambda", lambda_0_default)

        self.current_epoch = 0
        self.metrics = np.zeros(self.settings.max_iter + 1)
        reps = (self.settings.max_iter + 1, 1)
        self.xs = np.tile(self.x_0, reps)
        self.zus = np.tile(self.zu_0, reps)
        self.ys = np.tile(self.y_0, reps)
        self.lambdas = np.tile(self.lambda_0, reps)

        self.A_1 = np.zeros((self.N, self.N + self.M))
        for i in range(self.N):
            self.A_1[i, i] = -1

        self.data_dir_path = data_dir_path
        self.problem_name = problem_name

    def _validate_everything(self):
        self.validate()
        self.settings.validate()
        self._validate_initvals()

    def _validate_initvals(self):
        if self.x_0.shape != (self.N,):
            raise ValueError("Initial 'x_0' shape should be equal to amount of items")
        if self.y_0.shape != (self.N,):
            raise ValueError("Initial 'y_0' shape should be equal to amount of items")
        if self.lambda_0.shape != (self.N,):
            raise ValueError("Initial 'lambda_0' shape should be equal to amount of items")
        if self.zu_0.shape != (self.N + self.M,):
            raise ValueError(
                "Initial 'zu_0' shape should be equal to amount of items plus amount of objectives dimensions")

    def _profit(self, epoch):
        return self.xs[epoch].T.dot(self.profits.dot(self.xs[epoch]))

    def _loss_due_out_of_sync(self, epoch):
        return self.settings.mu * self.loss(self.xs[epoch], self.zus[epoch])

    def _calculate_metric(self, epoch):
        return -self._profit(epoch) + self._loss_due_out_of_sync(epoch)

    def _difference_for_xs_zus(self, x_epoch, zu_epoch):
        return self.xs[x_epoch] + self.A_1.dot(self.zus[zu_epoch])

    def _initialize_solving(self):
        self._validate_everything()
        self.metrics[0] = self._calculate_metric(0)

    def _save_metrics_data(self):
        if self.data_dir_path is None or self.problem_name is None:
            return
        file_path = os.path.join(self.data_dir_path, self.problem_name)
        dump_data = {
            "x": self.xs.tolist(),
            "y": self.ys.tolist(),
            "lambda": self.lambdas.tolist(),
            "zu": self.zus.tolist(),
            "metrics": self.metrics.tolist(),
        }
        with open(file_path, "w", encoding='utf-8') as result_file:
            result_file.write(json.dumps(dump_data))

    def solve(self) -> np.ndarray:
        """
        Solve QMdMCKP using admm with 3 block
        :return: Array of assignments
        """
        self._initialize_solving()
        for self.current_epoch in range(1, self.settings.max_iter + 1):
            self._solving_step()

        best_epoch = self.metrics.argmin()

        self._save_metrics_data()

        return self.xs[best_epoch]

    def update_settings(self, epoch) -> Settings:
        buffer = dict()
        rho = self.settings.rho
        alpha = self.settings.alpha
        beta = self.settings.beta
        gamma = self.settings.gamma
        mu = self.settings.mu

        increase = 2
        decrease = 2
        threshold = 10

        buffer["rho"] = rho
        buffer["alpha"] = alpha
        buffer["beta"] = beta
        buffer["gamma"] = gamma
        buffer["mu"] = mu

        for key, c in buffer.items():
            s = c * self.A_1 @ (self.zus[epoch] - self.zus[epoch - 1])
            norm_of_s = np.linalg.norm(s)
            r = self._difference_for_xs_zus(epoch, epoch)
            norm_of_r = np.linalg.norm(r)
            if norm_of_r > threshold * norm_of_s:
                buffer[key] *= increase
            elif norm_of_s > threshold * norm_of_r:
                buffer[key] /= decrease

        new_settings = Settings()

        for key, value in new_settings.__dict__:
            if buffer.__contains__(key):
                new_settings.__setattr__(key, buffer[key])
            else:
                new_settings.__setattr__(key, new_settings.__getattribute__(key))

        return new_settings

    def _solving_step(self):
        # Qubo block
        x_epoch = self.current_epoch - 1
        zu_epoch = self.current_epoch - 1
        y_epoch = self.current_epoch - 1
        lambda_epoch = self.current_epoch - 1
        self.xs[self.current_epoch] = self._calculate_x(x_epoch, zu_epoch, y_epoch, lambda_epoch)
        x_epoch += 1
        # Convex block
        self.zus[self.current_epoch] = self._calculate_zu(x_epoch, zu_epoch, y_epoch, lambda_epoch)
        zu_epoch += 1
        # Convex + quadratic block
        self.ys[self.current_epoch] = self._calculate_y(x_epoch, zu_epoch, y_epoch, lambda_epoch)
        y_epoch += 1
        # update lambda
        self.lambdas[self.current_epoch] = self._calculate_lambda(x_epoch, zu_epoch, y_epoch, lambda_epoch)
        lambda_epoch += 1
        # calculate profits
        self.metrics[self.current_epoch] = self._calculate_metric(self.current_epoch)
        # update settings
        self.settings = self.update_settings(self.current_epoch)

    # TODO make it pretty
    def _from_adjacency_matrix_to_edge_dict(self, matrix: np.ndarray, eps):
        if len(matrix.shape) != 2:
            raise ValueError("Argument is not matrix")
        N = matrix.shape[0]
        if matrix.shape != (N, N):
            raise ValueError("Matrix is not square")
        edge_dict = {}
        for i in range(N):
            for j in range(N):
                if abs(matrix[i, j]) > eps:
                    edge_dict[(i, j)] = matrix.item((i, j))
        return edge_dict

    # TODO make it pretty
    def _qubo_solver(self, Q: np.ndarray, eps: np.float64):
        qubo_sampler = SimulatedAnnealingSampler()
        edges_dict = self._from_adjacency_matrix_to_edge_dict(Q, eps)
        sampled = qubo_sampler.sample_qubo(edges_dict)
        solution = sampled.first.sample
        size = max(solution.keys()) + 1
        result = np.zeros(size)
        for key, item in solution.items():
            result[key] = item
        return result

    # TODO make it pretty
    def _calculate_x(self, x_epoch, zu_epoch, y_epoch, lambda_epoch):
        qubo_matrix = - self.profits + self.settings.alpha / 2 * self.groups.T.dot(self.groups) + self.settings.rho / 2 * np.eye(self.N)
        diagonal = - (self.settings.alpha * self.groups.T.dot(np.ones(self.K)) + self.settings.rho * (self.A_1.dot(self.zus[zu_epoch]) + self.ys[y_epoch]) - self.lambdas[
            lambda_epoch])
        qubo_matrix += np.diag(diagonal)
        # adding penalty beta / 2 * \| Wx + u - c \|^2
        # beta / 2 * |Wx + u - c|^T |Wx + u - c|
        # beta / 2 * (x^TW^TWx + 2(u-c)^TWx)
        qubo_matrix += self.settings.beta / 2 * self.weights.T * self.weights
        qubo_matrix += self.settings.beta * np.diag((self.zus[zu_epoch][self.N:] - self.capacities).T.dot(self.weights))
        # Qubo is list of edges with weights
        return self._qubo_solver(qubo_matrix, self.settings.eps)

    # TODO make it pretty
    def _calculate_zu(self, x_epoch, zu_epoch, y_epoch, lambda_epoch):
        q = self.lambdas[lambda_epoch] + self.settings.rho * (self.xs[x_epoch] - self.ys[y_epoch])
        cones = [clarabel.NonnegativeConeT(self.M)]
        matrix = csc_matrix(self.settings.rho * np.eye(self.N))
        weights_csc = csc_matrix(self.weights)
        solver = clarabel.DefaultSolver(matrix, q, weights_csc, self.capacities, cones, self.clarabel_settings)
        solution = solver.solve()
        if not solution.status:
            raise ValueError("Solution not found for convex block")
        return np.hstack([solution.x, solution.z])

    def _calculate_y(self, x_epoch, zu_epoch, y_epoch, lambda_epoch):
        difference = self._difference_for_xs_zus(x_epoch, zu_epoch)
        numerator = self.lambdas[lambda_epoch] + self.settings.rho * difference
        denominator = self.settings.gamma + self.settings.rho
        return numerator / denominator

    def _calculate_lambda(self, x_epoch, zu_epoch, y_epoch, lambda_epoch):
        difference = self._difference_for_xs_zus(x_epoch, zu_epoch) - self.ys[y_epoch]
        previous_lambda = self.lambdas[lambda_epoch]
        total = previous_lambda + self.settings.rho * difference
        return total

