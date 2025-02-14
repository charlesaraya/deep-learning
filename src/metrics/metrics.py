import numpy as np
from abc import ABC, abstractmethod
from typing import Literal

class Metric:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, y_hat, y_target):
        pass

SCOPES = {'batch', 'training', 'validation', 'test'}

class MetricManager:
    def __init__(self, metrics=None):
        self.metrics: list[Metric] = []
        self.results = {}
        # Init metrics
        if isinstance(metrics, str):
            metrics = [metrics]
        for metric in metrics:
            if isinstance(metric, str) and metric in METRICS:
                self.metrics.append(METRICS[metric]())
            elif issubclass(metric.__class__, Metric):
                self.metrics.append(metric)
            else:
                raise ValueError(f"Unsupported metric: {metric}.")
        # Init metric results dict
        self.init_results()

    def init_results(self, scope: str = None) -> None:
        """Clear stored metric results."""
        if not scope:
            # Init all scopes
            self.results = {
                'batch': {metric.name: [] for metric in self.metrics},
                'training': {metric.name: [] for metric in self.metrics},
                'validation': {metric.name: [] for metric in self.metrics},
                'test': {metric.name: [] for metric in self.metrics},
            }
        if scope in SCOPES:
            self.results[scope] = {metric.name: [] for metric in self.metrics}

    def compute(self, y_hat, y_target, scope: str):
        """Compute compiled metrics and store results."""
        if scope not in SCOPES:
            raise ValueError(f"Incorrect metric scope: {scope}.")
        for metric in self.metrics:
            result = metric.compute(y_hat, y_target)
            self.results[scope][metric.name].append(result)
        return self.results[scope]

    def get_latest_result(self, scope: str, metric_name: str):
        if scope not in SCOPES:
            raise ValueError(f"Incorrect metric scope: {scope}.")
        if metric_name not in METRICS:
            raise ValueError(f"Incorrect metric name: {metric_name}.")
        if not self.results[scope][metric_name]:
            return 0.0
        else:
            return self.results[scope][metric_name][-1]

class Accuracy(Metric):
    def __init__(self, name: str = 'accuracy'):
        super(Accuracy, self).__init__(name)

    def compute(self, y_hat, y_target):
        predictions = np.argmax(y_hat, axis=1)
        accuracy = np.mean(predictions == np.argmax(y_target, axis=1))
        return accuracy

class R2Score(Metric):
    """Measures how well the model explains variance in the data"""
    def __init__(self, name: str = 'r2_score'):
        super(R2Score, self).__init__(name)

    def compute(self, y_hat, y_target):
        numerator = np.sum((y_hat - y_target)**2)
        denominator = np.sum((y_target - np.mean(y_target))**2)
        r2_score = 1 - numerator/denominator
        return r2_score

class MeanSquaredError(Metric):
    """Squares the errors before averaging, giving more weight to larger errors."""
    def __init__(self, name: str = 'mse'):
        super(MeanSquaredError, self).__init__(name)

    def compute(self, y_hat, y_target):
        mse = np.mean((y_hat - y_target)**2)
        return mse

class MeanAbsoluteError(Metric):
    """Measures the average absolute difference between predictions and actual values."""
    def __init__(self, name: str = 'mae'):
        super(MeanAbsoluteError, self).__init__(name)

    def compute(self, y_hat, y_target):
        mse = np.mean(np.abs(y_hat - y_target))
        return mse

METRICS = {
    'accuracy': Accuracy,
    'r2_score': R2Score,
    'mse': MeanSquaredError,
    'mae': MeanAbsoluteError,
    None: None,
}