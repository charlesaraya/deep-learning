import numpy as np

from layers.activations import ACTIVATION_FN
from layers.layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_size: int, output_size: int, activation: None | str = None):
        super(DenseLayer, self).__init__(input_size, output_size)
        
        # Initiliaze weights and bias
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        # Set activation function
        self.activation = ACTIVATION_FN[activation] if activation else None

    def forward(self, input_data: np.ndarray):
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