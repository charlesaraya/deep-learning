import numpy as np
from typing import Literal

from layers.activations import ACTIVATION_FN
from layers.layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_size: int, output_size: int, weight_init: str = Literal['random', 'xavier', 'he'], activation: None | str = None):
        super(DenseLayer, self).__init__(input_size, output_size)
        
        # Initiliaze weights and bias
        self.weights = self.init_weight(weight_init)
        self.bias = np.zeros((1, output_size))
        # Set activation function
        self.activation = ACTIVATION_FN[activation] if activation else None

    def init_weight(self, weight_init):
        """Initiliase Weights using a given strategy"""
        match weight_init:
            case 'random':
                return np.random.randn(self.shape[0], self.shape[1]) * 0.01
            case 'xavier':
                upper = np.sqrt(1.0 / self.shape[0])
                lower = -upper
                return np.random.uniform(lower, upper, self.shape)
            case 'he':
                return np.random.randn(self.shape[0], self.shape[1]) * np.sqrt(2 / self.shape[0])

    def forward(self, input_data: np.ndarray, is_training: bool = True):
        """Forward pass"""
        self.input = input_data
        # Linear Transform
        self.output = np.dot(self.input, self.weights) + self.bias
        # Activation Layer
        self.output = self.activation(self.output) if self.activation else self.output
        return self.output

    def backward(self, output_gradient: np.ndarray):
        """Backward pass"""
        doutput = output_gradient * (self.activation(self.output, derivative=True) if self.activation else 1)

        # Gradients for weights and bias
        self.dweights = np.dot(self.input.T, doutput)
        self.dbias = np.sum(doutput, axis=0, keepdims=True)

        # Gradient to be passed to the previous layer
        dinput = np.dot(doutput, self.weights.T)

        return dinput