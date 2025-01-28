import numpy as np
from abc import ABC, abstractmethod

class Layer:
    """Layer base class.
    """
    def __init__(self, input_size: int = 0, output_size: int = 0, name: str = None, uid: int = None):
        self.shape = (input_size, output_size)

        self.name = name if name is not None else self.__class__.__name__
        self.name = str.lower(name)
        if uid is not None:
            self.name = f"{self.name}_{uid}"

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