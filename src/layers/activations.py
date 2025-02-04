import numpy as np

from layers.layer import Layer

class Sigmoid(Layer):
    """Applies Tanh activation function to the input.

    The Tanh activation function maps input values to the range (0, 1).
    Mostly used in in shallow networks , binary classification problems.
    Some drawbacks are vanishing gradients during backpropagation from deeper hidden layers to inputs, 
    gradient saturation, and slow convergence. After numerous iterations the value of gradient are so small
    that the weights get updated very slowly.
    """
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

    def forward(self, Z: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Performs the forward pass through the layer.

        #### Args
            - `Z` (`np.ndarray`): The input array, typically a pre-activation value (logits) from a layer.

        #### Returns
            - `np.ndarray`: The activation values after applying the activation function.
        """
        self.h = 1. / (1. + np.exp(-Z))
        return self.h

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the layer.

        This method computes the gradients of the input, and propagates them to the previous layer in the network.

        #### Args
            - `dloss` (`np.ndarray`): The gradient of the loss w.r.t. the next layer's output.

        #### Returns
            - `np.ndarray`: The gradient of the loss w.r.t. the layer's input.
        """
        return dloss * (self.h * (1. - self.h))

class Tanh(Layer):
    """Implements a Tanh (Hyperbolic Tangent) activation layer in a neural network.

    The Tanh activation function maps input values to the range (-1, 1), useful for dealing with negative values more effectively. 
    Preferred over Sigmoid as it gives better performance for multi-layer neural networks. 
    Does not solve the vanishing gradient problem that sigmoids suffers.
    """
    def __init__(self, **kwargs):
        super(Tanh, self).__init__(**kwargs)

    def forward(self, Z: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Performs the forward pass through the layer.

        #### Args
            - `Z` (`np.ndarray`): The input array, typically a pre-activation value (logits) from a layer.
            - `is_training` (`bool`, optional): Indicates whether the forward pass is being performed during training or inference (default = True).

        #### Returns
            - `np.ndarray`: The activation values after applying the activation function.
        """
        self.h = np.tanh(Z)
        return self.h

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the layer.

        This method computes the gradients of the input, and propagates them to the previous layer in the network.

        #### Args
            - `dloss` (`np.ndarray`): The gradient of the loss w.r.t. the next layer's output.

        #### Returns
            - `np.ndarray`: The gradient of the loss w.r.t. the layer's input.
        """
        return dloss * (1. - np.tanh(self.h)**2)

class ReLU(Layer):
    """Implements a (Leaky) ReLU (Rectified Linear Unit) activation layer in a neural network.

    ReLU is a type of activation function that are linear in the positive dimension, but zero in the negative dimension.
    Eliminates the vanishing gradient problem by rectifying the values of the inputs less than zero and forcing them to zero.
    Linearity in the positive dimension has the attractive property that it prevents non-saturation of gradients.
    Guarantee faster computation since it does not compute exponentials and divisions, with overall speed of computation enhanced.
    Sometimes fragile during training causing some of the gradients to die, leading to dying neurons.

    Leaky ReLU has a small slope for negative values instead of a flat slope.
    To resolve the dead neuron issues in tasks that may suffer from sparse gradients, the leaky ReLU was proposed with a small 
    negative slope to the ReLU to sustain and keep the weight updates alive during the entire propagation process.
    """
    def __init__(self, alpha: float = 0, **kwargs):
        """Initiliases ReLU layer with optional alpha to turn it into Leaky ReLU

        #### Args
            - `alpha` (`float`, optional): The slope of the function for negative inputs.
                - `0`(default): Performs regular ReLU activation.
                - `alpha` >`0`: Performs Leaky ReLU activation.
        """
        super(ReLU, self).__init__(**kwargs)
        self.alpha = alpha

    def forward(self, Z: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Performs the forward pass through the layer.

        #### Args
            - `Z` (`np.ndarray`): The input array, typically a pre-activation value (logits) from a dense or convolutional layer.

        #### Returns
            - `np.ndarray`: The activation values after applying the activation function.
        """
        self.h = np.maximum(self.alpha * Z, Z)
        return self.h

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the layer.

        This method computes the gradients of the input, and propagates them to the previous layer in the network.

        #### Args
            - `dloss` (`np.ndarray`): The gradient of the loss w.r.t. the next layer's output.

        #### Returns
            - `np.ndarray`: The gradient of the loss w.r.t. the layer's input.
        """
        return dloss * np.where(self.h <= 0, self.alpha, 1.)

class SoftMax(Layer):
    """Implements a Softmax activation layer in a neural network.
    
    The softmax function converts logits (raw scores) into a probability distribution, 
    ensuring that the output values are in the range [0, 1] and sum to 1 across each sample.
    """
    def __init__(self, **kwargs):
        super(SoftMax, self).__init__(**kwargs)

    def forward(self, Z: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Performs the forward pass through the layer.

        #### Args
            - `Z` (`np.ndarray`): The input array of pre-activated logits, typically from a dense layer.

        #### Returns
            - `np.ndarray`: The activation values after applying the activation function.
        """
        exp_x = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        y_hat = exp_x / np.sum(exp_x, axis=1, keepdims=True) # predicted probability for each class
        return y_hat

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        return dloss # for now, calculation done outside as it's tied with cross-entropy loss

ACTIVATIONS = {
    'sigmoid': Sigmoid(),
    'relu': ReLU(),
    'tanh': Tanh(),
    'softmax': SoftMax(),
    None: None
}