"""Script for time measurement experiments on linear regression models."""
import argparse
from typing import List
from typing import Tuple
from typing import Type

import pandas as pd

import lr


def get_args() -> argparse.Namespace:
    """Parses script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets-dir',
        required=True,
        help='Name of directory with generated datasets',
        type=str,
    )

    return parser.parse_args()


def run_experiments(
    models: List[Type[lr.base.LinearRegression]],
    datasets: List[Tuple[List[float], List[float]]],
) -> pd.DataFrame:
    pass


def make_plot(results: pd.DataFrame) -> None:
    pass


def main() -> None:
    """Runs script."""
    args = get_args()

    models = [
        lr.LinearRegressionNumpy,
        lr.LinearRegressionProcess,
        lr.LinearRegressionSequential,
        lr.LinearRegressionThreads,
    ]

    datasets = []  # TODO: read datasets

    results = run_experiments(models, datasets)

    make_plot(results)


if __name__ == '__main__':
    main()
