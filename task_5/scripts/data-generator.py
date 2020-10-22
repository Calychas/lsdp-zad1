"""Script for generation of artificial datasets."""
import argparse
from typing import List
from typing import Tuple
import numpy as np
import pickle as pkl
import os

def get_args() -> argparse.Namespace:
    """Parses script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num-samples',
        required=True,
        help='Number of samples to generate',
        type=int,
    )
    parser.add_argument(
        '--out-dir',
        required=True,
        help='Name of directory to save generated data',
        type=str,
    )

    return parser.parse_args()


def generate_data(num_samples: int) -> Tuple[List[float], List[float]]:
    """Generated X, y with given number of data samples."""
    X = list(np.linspace(0, 100, num=num_samples))
    y = list(map(lambda x: 5 * x + 100 + np.random.uniform(-40, 40), X))

    return X, y


def main() -> None:
    """Runs script."""
    args = get_args()
    num_samples = args.num_samples
    out_dir = args.out_dir
    data = generate_data(num_samples)

    pkl.dump(data, open(os.path.join(os.path.normpath(out_dir), f"data_{num_samples}.pkl"), "wb"))

if __name__ == '__main__':
    main()
