from lr import base
from concurrent.futures import Executor
import numpy as np
from typing import Tuple, List
import abc

import multiprocessing

class LinearRegressionConcurrent(base.LinearRegression):

    @abc.abstractmethod
    def get_executor(self, max_workers) -> Executor:
        pass

    def task(self, data: Tuple[np.ndarray, np.ndarray]):
        X, y = data
        sum_x = X.sum()
        sum_y = y.sum()
        sum_x2 = np.sum(X ** 2)
        sum_xy = np.sum(X * y)

        return np.array([sum_x, sum_y, sum_x2, sum_xy])

    def fit(self, X: List[float], y: List[float]) -> base.LinearRegression:
        X = np.array(X)
        y = np.array(y)
        n = X.shape[0]

        n_workers = multiprocessing.cpu_count()
        data_chunked = zip(np.array_split(X, n_workers), np.array_split(y, n_workers))
        ex = self.get_executor(max_workers=n_workers)
        tasks_scheduled = ex.map(self.task, data_chunked)
        results = list(tasks_scheduled)
        results_summed = np.sum(np.stack(results), axis=0)

        sum_x = results_summed[0]
        sum_y = results_summed[1]
        sum_x2 = results_summed[2]
        sum_xy = results_summed[3]

        denominator = n * sum_x2 - sum_x**2
        a = (n * sum_xy - sum_x * sum_y) / denominator
        b = (sum_y * sum_x2 - sum_x * sum_xy) / denominator
        self._coef = (b, a)

        return self

