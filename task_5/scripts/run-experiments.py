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
from timeit import default_timer as timer
import plotly.express as px
from tqdm.auto import tqdm

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

    cols = ["name", "size", "time"]
    experiment_data = []
    for dataset in tqdm(datasets):
        X, y = dataset
        n = len(X)
        X_train, y_train = X, y
        for model_cons in models:
            model = model_cons()
            model_name = model.get_name()

            start = timer()
            model.fit(X_train, y_train)
            end = timer()
            time_elapsed = end - start

            experiment_data.append([model_name, n, time_elapsed])

    return pd.DataFrame(data=experiment_data, columns=cols)


def save_results(results: pd.DataFrame) -> None:
    results.to_csv("../results/results.csv")

def make_plot(results: pd.DataFrame) -> None:
    fig = px.line(results, x="size", y="time", color="name")
    fig.write_html("../results/results.html")
    fig.show(renderer="firefox")


def main() -> None:
    """Runs script."""
    random.seed(42)

    args = get_args()
    datasets_dir = args.datasets_dir
    file_paths = list(map(lambda p: os.path.join(datasets_dir, p), os.listdir(datasets_dir)))
    datasets = get_datasets(file_paths)


    models = [
        lr.LinearRegressionSequential,
        lr.LinearRegressionNumpy,
        lr.LinearRegressionThreads,
        lr.LinearRegressionProcess,
    ]

    results = run_experiments(models, datasets)
    results.sort_values("size", axis=0, inplace=True)
    save_results(results)
    make_plot(results)


if __name__ == '__main__':
    main()
