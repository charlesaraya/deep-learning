import numpy as np
from abc import ABC, abstractmethod

class Loss:
    """Loss base class.
    """
    def __init__(self):
        pass

    @abstractmethod
    def forward(y_hat: np.ndarray, y: np.ndarray) -> float:
        """Forward pass"""
        pass

    @abstractmethod
    def backward(y_hat: np.ndarray, y: np.ndarray) -> float:
        """Backward pass"""
        pass