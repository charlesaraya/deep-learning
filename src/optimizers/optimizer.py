import numpy as np

class Optimizer:
    """Optimizer base class."""
    def __init__(self, name:str = None):
        self.name = name

    def init_params(self, trainable_params: np.ndarray):
        """Initiliaze internal parameters."""
        pass

    def update(self, gradient: list):
        """Performs an update pass through the optimizer."""
        pass