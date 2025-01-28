import numpy as np
from typing import Literal

from layers.activations import ACTIVATION_FN
from layers.layer import Layer

class Dense(Layer):
    """Implements a dense fully-conected layer in a neural network, characterized by a linear 
        transformation followed by an optional activation function.
    """
    def __init__(
        self,
        shape: tuple,
        weight_init: str = Literal['random', 'xavier', 'he'],
        activation: None | str = None
    ):
        """Initializes the Dense (fully connected) layer.

        The layer's weights can be initialized using different strategies to improve convergence and training performance.

        ### Args
            - `shape` (`tuple`): A tuple defining the structure of the layer as (input_size, output_size), where:
                - `input_size`: The number of features in the input data.
                - `output_size`: The number of neurons in the layer.
            - weight_init (str, optional): Specifies the weight initialization strategy:
                - `'random'` (default): Initializes weights with small random values.
                - `'xavier'`: Uses Xavier/Glorot initialization, suitable for tanh/sigmoid.
                - `'he'`: Uses He initialization, ideal for ReLU/Leaky ReLU activations
            - activation (None | str, optional): The activation function to be applied after the linear transformation. 
                Pass `None` for no activation (default = None).
        """
        super(Dense, self).__init__(*shape)
        # Initiliaze weights and bias
        self.weights = self.init_weight(weight_init)
        self.bias = np.zeros((1, self.shape[1]))
        # Set activation function
        self.activation = ACTIVATION_FN[activation] if activation else None

    def init_weight(self, weight_init):
        """Initializes the weights of a layer using the specified initialization strategy.

        This method supports several common weight initialization techniques, each tailored 
        to improve model performance and stability during training.

        #### Args
            - `weight_init` (`str`): The strategy for initializing weights. Options include:
                - `'random'`: Initializes weights with small random values scaled by 0.01.
                - `'xavier'`: Uses the Xavier/Glorot uniform initialization, suitable for layers with sigmoid or tanh activations.
                - `'he'`: Uses He initialization, ideal for layers with ReLU or Leaky ReLU activations.
        #### Returns
            - `np.ndarray`: The initialized weights with the same shape as the layer.
        """
        match weight_init:
            case 'random':
                return np.random.randn(*self.shape) * 0.01
            case 'xavier':
                upper = np.sqrt(1.0 / self.shape[0])
                lower = -upper
                return np.random.uniform(lower, upper, self.shape)
            case 'he':
                return np.random.randn(*self.shape) * np.sqrt(2 / self.shape[0])

    def forward(self, input_data: np.ndarray, is_training: bool = True):
        """Performs a forward pass through the layer.

        #### Args
            - `input_data` (`np.ndarray`): The input data for the layer, typically a 2D array with shape (batch_size, input_features).
            - `is_training` (`bool`, optional): Indicates whether the forward pass is being performed during training or inference (default = True).

        #### Returns
            - `np.ndarray`: The output of the layer after applying the linear transformation.
        """
        self.input = input_data
        # Linear Transform
        self.output = np.dot(self.input, self.weights) + self.bias
        # Activation Layer
        self.output = self.activation(self.output) if self.activation is not None else self.output
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the layer.

        This method computes the gradients of the loss w.r.t. the layer's weights, biases, and input. 
        It propagates the gradient to the previous layer in the network.

        #### Args
            - `output_gradient` (`np.ndarray`): The gradient of the loss w.r.t. the next layer's output.

        #### Returns
            - `np.ndarray`: The gradient of the loss w.r.t. the layer's input.
        """
        doutput = output_gradient * (self.activation(self.output, derivative=True) if self.activation is not None else 1)

        # Gradients for weights and bias
        self.dweights = np.dot(self.input.T, doutput)
        self.dbias = np.sum(doutput, axis=0, keepdims=True)

        # Gradient to be passed to the previous layer
        dinput = np.dot(doutput, self.weights.T)

        return dinput

    def update(self, learning_rate: float) -> None:
        """Updates the layer's parameters (weights and biases) using the computed gradients.

        This method applies gradient descent to adjust the weights and biases of the layer, 
        minimizing the loss function during training.

        #### Args
            - `learning_rate` (`float`): The learning rate used to scale the gradient updates. 

        #### Returns
            - `None`: Updates the Layer's internal prameters and returns.
        """
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias
        return None