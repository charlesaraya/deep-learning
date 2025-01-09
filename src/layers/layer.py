import numpy as np
from abc import ABC, abstractmethod

class Layer:
    def __init__(self, input_size: int, output_size: int):
        self.shape = (input_size, output_size)

    @abstractmethod
    def forward(self, input: np.ndarray):
        """Forward pass"""
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray):
        """Backward pass"""
        pass