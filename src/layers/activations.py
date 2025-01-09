import numpy as np

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

ACTIVATION_FN = {
    'sigmoid': sigmoid_activation,
    'relu': relu_activation,
    'tanh': tanh_activation,
    'softmax': softmax_activation,
    None: None
}