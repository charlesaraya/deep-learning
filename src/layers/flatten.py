import numpy as np

from layers.layer import Layer

class Flatten(Layer):
    def __init__(self, **kwargs):
        super(Flatten, self).__init__(**kwargs)

    def forward(self, input_data: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Performs a forward pass through the layer.

        Reshapes the input for the next layer (From PoolLayer to DenseLayer).
        """
        self.prev_shape = input_data.shape
        flattened_shape = np.prod(self.prev_shape[1:])
        return input_data.reshape(input_data.shape[0], flattened_shape)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Performs a backward pass through the layer.

        Reshapes the gradients for the previous layer (From DenseLayer to PoolLayer)
        """
        return output_gradient.reshape(self.prev_shape)