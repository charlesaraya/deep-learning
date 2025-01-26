import numpy as np
from abc import ABC, abstractmethod

class Layer:
    def __init__(self, input_size: int = 0, output_size: int = 0):
        self.shape = (input_size, output_size)

    @abstractmethod
    def forward(self, input: np.ndarray, is_training: bool = True):
        """Performs a forward pass through the layer."""
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray):
        """Performs a backward pass through the layer."""
        pass

    def update(self, learning_rate: float):
        """Performs an update pass tghrough the layer."""
        pass