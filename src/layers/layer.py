import numpy as np
from abc import ABC, abstractmethod

class Layer:
    """Layer base class.
    """
    def __init__(self, shape: tuple[int] = None, name: str = None, uid: int = None):
        self.shape = shape

        self.name = name if name is not None else self.__class__.__name__
        self.name = str.lower(self.name)
        if uid is not None:
            self.name = f"{self.name}_{uid}"

        self.trainable_params: list = None
        self.total_params = 0
        self.gradients: list = None

    def set_trainable_params(self, *params) -> None:
        self.trainable_params = []
        for param in params:
            self.trainable_params.append(param)
            self.total_params += param.size
        return None

    def init_gradients(self) -> list[np.ndarray]:
        self.gradients = []
        for param in self.trainable_params:
            gradient = np.zeros_like(param)
            self.gradients.append(gradient)
        return self.gradients

    @abstractmethod
    def forward(self, input: np.ndarray, is_training: bool = True):
        """Performs a forward pass through the layer."""
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray):
        """Performs a backward pass through the layer."""
        pass

    def update(self, learning_rate: float):
        """Performs an update pass through the layer."""
        pass