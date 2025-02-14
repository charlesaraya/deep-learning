import numpy as np

class Optimizer:
    """Optimizer base class."""
    def __init__(
        self,
        clip_gradient_value = None,
        name: str = None,
    ):
        self.name = name
        self.clip_gradient_value = clip_gradient_value

    def clip_gradients(self, gradients):
        if self.clip_gradient_value and self.clip_gradient_value > 0:
            return np.clip(gradients, -self.clip_gradient_value, self.clip_gradient_value)
        else:
            return gradients

    def init_params(self, trainable_params: np.ndarray):
        """Initiliaze internal parameters."""
        pass

    def update(self, gradient: list):
        """Performs an update pass through the optimizer."""
        pass