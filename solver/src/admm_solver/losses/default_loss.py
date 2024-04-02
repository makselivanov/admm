import numpy as np

from ..loss import Loss


class DefaultLoss(Loss):
    def __call__(self, x: np.ndarray, zu: np.ndarray):
        N = x.shape[0]
        NM = zu.shape[0]
        A_1 = np.zeros((N, NM))
        for i in range(N):
            A_1[i, i] = -1
        return ((x + A_1.dot(zu)) ** 2).sum()
