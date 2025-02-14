import numpy as np
from abc import ABC, abstractmethod

class Metric:
    def __init__(self, name: str):
        self.name = name
    @abstractmethod
    def compute_metric(self, y_hat, y_target):
        pass

class Accuracy(Metric):
    def __init__(self, name: str = 'accuracy'):
        super(Accuracy, self).__init__(name)

    def compute_metric(self, y_hat, y_target):
        predictions = np.argmax(y_hat, axis=1)
        accuracy = np.mean(predictions == np.argmax(y_target, axis=1))
        return accuracy

METRICS = {
    'accuracy': Accuracy,
    None: None
}