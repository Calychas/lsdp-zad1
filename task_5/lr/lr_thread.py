from lr.lr_conc import LinearRegressionConcurrent
from concurrent.futures import ThreadPoolExecutor, Executor


class LinearRegressionThreads(LinearRegressionConcurrent):
    def get_name(self) -> str:
        return "Thread"

    def get_executor(self, max_workers) -> Executor:
        return ThreadPoolExecutor(max_workers=max_workers)

