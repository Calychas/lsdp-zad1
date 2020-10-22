from concurrent.futures import Executor, ProcessPoolExecutor

from lr.lr_conc import LinearRegressionConcurrent


class LinearRegressionProcess(LinearRegressionConcurrent):
    def get_executor(self, max_workers) -> Executor:
        return ProcessPoolExecutor(max_workers=max_workers)
