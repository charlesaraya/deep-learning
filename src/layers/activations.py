import numpy as np

from layers.layer import Layer

def sigmoid_activation(Z: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Applies Sigmoid activation function to the input.

    The Sigmoid activation function maps input values to the range (0, 1), useful for 
    modeling probabilities. 

    Args:
        Z (ndarray): Input array, typically a pre-activation value (logits) from a layer.
        derivative (bool, optional): Computes the derivative of the Sigmoid function instead of the activation itself.

    Returns:
        ndarray: Output array with the activation value or derivative for the layer.
    """
    if derivative:
        return Z * (1. - Z)
    return 1. / (1. + np.exp(-Z))

class Sigmoid(Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.h = 1. / (1. + np.exp(-Z))
        return self.h

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        return dloss * (self.h * (1. - self.h))

def tanh_activation(Z: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Applies Tanh activation function to the input.

    The Sigmoid activation function maps input values to the range (-1, 1), useful for 
    dealing with negative values more effectively. 

    Args:
        Z (ndarray): Input array, typically a pre-activation value (logits) from a layer.
        derivative (bool, optional): Computes the derivative of the Tanh function instead of the activation itself.

    Returns:
        ndarray: Output array with the activation value or derivative for the layer.
    """
    tanh = np.divide(np.exp(Z) - np.exp(-Z), np.exp(Z) + np.exp(-Z)) # shortcut: np.tanh(Z)
    if derivative:
        return 1. - tanh**2
    return tanh

class Tanh(Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.h = np.tanh(Z)
        return self.h

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        return dloss * (1. - np.tanh(self.h)**2)

def relu_activation(Z: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Applies ReLU activation function to the input.

    A ReLU (Rectified Linear Unit) activation function is linear in the positive dimension, but zero in the negative dimension. 

    Args:
        Z (ndarray): Input array, typically a pre-activation value (logits) from a layer.
        derivative (bool, optional): Computes the derivative of the Relu function instead of the activation itself.

    Returns:
        ndarray: Output array with the activation value or derivative for the layer.
    """
    if derivative:
        return np.where(Z < 0, 0, 1.)
    return np.maximum(0, Z)

class ReLU(Layer):
    """Implements the (Leaky) ReLU activation function.
    """
    def __init__(self, alpha: float = 0):
        """Initiliases ReLU layer with optional alpha to turn it into Leaky ReLU

        Args:
            alpha (float, optional): The slope of the function for negative inputs. Defaults to 0 (ReLU).
        """
        super(ReLU, self).__init__()
        self.alpha = alpha

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.h = np.maximum(self.alpha * Z, Z)
        return self.h

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        return dloss * np.where(self.h < 0, self.alpha, 1.)

def softmax_activation(Z: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Applies Softmax activation function to the input.
    
    The softmax function converts logits (raw scores) into a probability distribution, 
    ensuring that the output values are in the range [0, 1] and sum to 1 across each sample.

    Args:
        Z (ndarray): Input array (logits). Each row corresponds to the raw scores for a single sample across all classes.

    Returns:
        ndarray: Output array with softmax probabilities. Each row represents a valid probability distribution.
    """
    if derivative:
        return 1
    exp_x = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    y_hat = exp_x / np.sum(exp_x, axis=1, keepdims=True) # predicted probability for each class
    return y_hat

class SoftMax(Layer):
    def __init__(self):
        super(SoftMax, self).__init__()

    def forward(self, Z: np.ndarray) -> np.ndarray:
        exp_x = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        y_hat = exp_x / np.sum(exp_x, axis=1, keepdims=True) # predicted probability for each class
        return y_hat

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        return dloss # for now, calculation done outside as it's tied with cross-entropy loss

ACTIVATION_FN = {
    'sigmoid': sigmoid_activation,
    'relu': relu_activation,
    'tanh': tanh_activation,
    'softmax': softmax_activation,
    None: None
}