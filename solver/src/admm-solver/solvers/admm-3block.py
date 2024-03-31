from abc import ABC

import clarabel
import numpy as np

from ..solver import AdmmSolver


class AdmmBlock3Solver(AdmmSolver, ABC):
    def solve(self) -> np.ndarray:
        profits = self.profit
        groups = self.groups
        weights = self.weights
        capacities = self.capacities

        clarabel_settings = clarabel.DefaultSettings()
        clarabel_settings.verbose = False
        N, M, K = self.validate()
        A_1 = np.zeros((N, N + M))
        for i in range(N):
            A_1[i, i] = -1
        self.settings.validate()
        if x_0 is None:
            x_0 = np.random.random_integers(0, 1, N)
        if zu_0 is None:
            zu_0 = np.random.rand(N + M)
        if y_0 is None:
            y_0 = np.random.rand(N)
        if lambda_0 is None:
            lambda_0 = np.random.rand(N)
        validateInitialValues(x_0, zu_0, y_0, lambda_0, N, M)
        epochs += 1
        x = np.tile(x_0, (epochs, 1))
        zu = np.tile(zu_0, (epochs, 1))
        y = np.tile(y_0, (epochs, 1))
        lamb = np.tile(lambda_0, (epochs, 1))
        metrics = np.zeros(epochs)
        metrics[0] = - x[0].T.dot(profits.dot(x[0])) + mu * loss(x[0], zu[0])
        for curr_epoch in range(1, epochs):
            prev_epoch = curr_epoch - 1
            # Qubo block
            qubo_matrix = - profits + alpha / 2 * groups.T.dot(groups) + rho / 2 * np.eye(N)
            diag = - (alpha * groups.T.dot(np.ones(K)) + rho * (A_1.dot(zu[prev_epoch]) + y[prev_epoch]) - lamb[prev_epoch])
            qubo_matrix += np.diag(diag)
            # adding penalty beta / 2 * \| Wx + u - c \|^2
            # beta / 2 * |Wx + u - c|^T |Wx + u - c|
            # beta / 2 * (x^TW^TWx + 2(u-c)^TWx)
            qubo_matrix += beta / 2 * weights.T * weights
            qubo_matrix += beta * np.diag((zu[prev_epoch][N:] - capacities).T.dot(weights))
            x[curr_epoch] = quboSolver(qubo_matrix, eps)
            # Qubo is list of edges with weights
            # Convex block
            q = lamb[prev_epoch] + rho * (x[curr_epoch] - y[prev_epoch])
            cones = [clarabel.NonnegativeConeT(M)]
            matrix = sparse.csc_matrix(rho * np.eye(N))
            weights_csc = sparse.csc_matrix(weights)
            solver = clarabel.DefaultSolver(matrix, q, weights_csc, capacities, cones, clarabel_settings)
            solution = solver.solve()
            if not solution.status:
                raise "Solution not found for convex block"
            zu[curr_epoch] = np.hstack([solution.x, solution.z])
            # Convex + quadratic block
            y[curr_epoch] = (lamb[prev_epoch] + rho * (x[curr_epoch] + A_1.dot(zu[curr_epoch]))) / (gamma + rho)
            # update lambda
            lamb[curr_epoch] = lamb[prev_epoch] + rho * (x[curr_epoch] + A_1.dot(zu[curr_epoch]) - y[curr_epoch])
            # calculate metrics
            metrics[curr_epoch] = - x[curr_epoch].T.dot(profits.dot(x[curr_epoch])) + mu * loss(x[curr_epoch],
                                                                                                zu[curr_epoch])

        best_epoch = metrics.argmin()
        return x[best_epoch]
