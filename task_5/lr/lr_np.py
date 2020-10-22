from typing import List

from lr import base
import numpy as np


class LinearRegressionNumpy(base.LinearRegression):
    def fit(self, X: List[float], y: List[float]) -> base.LinearRegression:
        X = np.array(X)[:, None]
        X = np.hstack((np.ones_like(X), X))
        y = np.array(y)[:, None]
        coef = ((np.linalg.inv(X.T.dot(X))).dot(X.T)).dot(y).squeeze()
        self._coef = (coef[0], coef[1])

        return self

