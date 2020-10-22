from typing import List

from lr import base


class LinearRegressionSequential(base.LinearRegression):

    def get_name(self) -> str:
        return "Sequential"

    def fit(self, X: List[float], y: List[float]) -> base.LinearRegression:
        n = len(X)
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_xy = 0

        for var_x, var_y in zip(X, y):
            sum_x += var_x
            sum_y += var_y
            sum_x2 += pow(var_x, 2)
            sum_xy += var_x * var_y

        denominator = n * sum_x2 - pow(sum_x, 2)
        a = (n * sum_xy - sum_x * sum_y) / denominator
        b = (sum_y * sum_x2 - sum_x * sum_xy) / denominator
        self._coef = (b, a)

        return self
