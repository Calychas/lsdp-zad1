"""Script for time measurement experiments on linear regression models."""
import argparse
from typing import List
from typing import Tuple
from typing import Type

import pandas as pd
import os
import pickle as pkl
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def get_datasets(file_paths: List[str]) -> List[Tuple[List[float], List[float]]]:
    datasets = list(map(lambda p: pkl.load(open(p, "rb")), file_paths))
    return datasets


def run_experiments(
    models: List[Type[lr.base.LinearRegression]],
    datasets: List[Tuple[List[float], List[float]]],
) -> pd.DataFrame:

    for dataset in datasets:
        X, y = dataset
        n = len(X)
        indicies = list(range(n))
        random.shuffle(indicies)
        split_index = math.ceil(len(X) * 0.8)

        X_shuffled, y_shuffled = (list(np.array(X)[indicies]), list(np.array(y)[indicies]))

        X_train, X_val = (X_shuffled[:split_index], X_shuffled[split_index:])
        y_train, y_val = (y_shuffled[:split_index], y_shuffled[split_index:])
        for model_cons in models:
            model = model_cons()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            sns.scatterplot(x=X_val, y=y_val)
            sns.scatterplot(x=X_val, y=y_pred)
            plt.show()

    return pd.DataFrame()

def make_plot(results: pd.DataFrame) -> None:
    pass


def main() -> None:
    """Runs script."""
    random.seed(42)

    args = get_args()
    datasets_dir = args.datasets_dir
    file_paths = list(map(lambda p: os.path.join(datasets_dir, p), os.listdir(datasets_dir)))
    datasets = get_datasets(file_paths[:1])  # FIXME


    models = [
        # lr.LinearRegressionSequential,
        lr.LinearRegressionNumpy,
        # lr.LinearRegressionProcess,
        # lr.LinearRegressionThreads,
    ]

    results = run_experiments(models, datasets)

    make_plot(results)


if __name__ == '__main__':
    main()
