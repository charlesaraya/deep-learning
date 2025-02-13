import numpy as np
from typing import Literal

from layers.activations import ACTIVATIONS
from layers.layer import Layer

class Recurrent(Layer):
    """Implements a recurrent layer in a neural network.
    """
    def __init__(
        self,
        shape: tuple,
        weight_init: str = Literal['random', 'xavier', 'he'],
        activation = None,
        **kwargs
    ):
        """Initializes the Recurrent layer.

        The layer's weights can be initialized using different strategies to improve convergence and training performance.

        ### Args
            - `shape` (`tuple`): A tuple defining the structure of the layer, specified as (input_size, output_size).
            - weight_init (str, optional): Specifies the weight initialization strategy:
                - `'random'` (default): Initializes weights with small random values.
                - `'xavier'`: Uses Xavier/Glorot initialization, suitable for tanh/sigmoid.
                - `'he'`: Uses He initialization, ideal for ReLU/Leaky ReLU activations
            - activation: The activation function to be applied after the linear transformation (default = None).
        """
        super(Recurrent, self).__init__(shape, **kwargs)

        # Initiliaze weights and bias
        self.weights = self.init_weight(shape, weight_init)
        self.prev_weights = self.init_weight((shape[1], shape[1]), weight_init)
        self.bias = np.zeros((1, self.shape[1]))

        self.hidden_state = None

        self.set_trainable_params(self.weights, self.prev_weights, self.bias)
        self.dweights, self.prev_weights, self.dbias = self.init_gradients()

        # Set activation function
        self.activation: Layer = ACTIVATIONS[activation]

    def init_weight(self, shape, weight_init):
        """Initializes the weights of a layer using the specified initialization strategy.

        This method supports several common weight initialization techniques, each tailored 
        to improve model performance and stability during training.

        #### Args
            - `shape` (`tuple[int]`): the shape of the weights matrix.
            - `weight_init` (`str`): The strategy for initializing weights. Options include:
                - `'random'`: Initializes weights with small random values scaled by 0.01.
                - `'xavier'`: Uses the Xavier/Glorot uniform initialization, suitable for layers with sigmoid or tanh activations.
                - `'he'`: Uses He initialization, ideal for layers with ReLU or Leaky ReLU activations.
        #### Returns
            - `np.ndarray`: The initialized weights with the same shape as the layer.
        """
        match weight_init:
            case 'random':
                return np.random.randn(*shape) * 0.01
            case 'xavier':
                upper = np.sqrt(1.0 / shape[0])
                lower = -upper
                return np.random.uniform(lower, upper, shape)
            case 'he':
                return np.random.randn(*shape) * np.sqrt(2 / shape[0])

    def forward(self, input_data: np.ndarray, is_training: bool = True):
        """Performs a forward pass through the layer.

        #### Args
            - `input_data` (`np.ndarray`): The input data for the layer, typically a 2D array with shape (batch_size, input_features).
            - `state` (`np.ndarray`): The hidden state from the previous time step.
            - `is_training` (`bool`, optional): Indicates whether the forward pass is being performed during training or inference (default = True).

        #### Returns
            - `np.ndarray`: The output of the layer after applying the linear transformation.
        """
        self.input = input_data
        # Initial state with shape: (batch_size, num_hiddens)
        batch_size, sequence_length, input_dim = self.input.shape
        _, hidden_dim = self.shape
        if self.hidden_state is None:
            self.hidden_state = np.zeros((batch_size, hidden_dim))

        output = []
        # Shape of inputs: (batch_size, sequence_length, input_dim)
        for t_step in range(sequence_length):
            input_t = self.input[:,t_step,:] # Shape: (batch_size, input_dim)
            input_t_hat = np.dot(input_t, self.weights) + np.dot(self.hidden_state, self.prev_weights) + self.bias
            self.hidden_state = self.activation.forward(input_t_hat)
            output.append(self.hidden_state)

        # Shape: (sequence_length, batch_size, hidden_dim)
        output = np.stack(output, axis=0)
        # Shape: (batch_size, sequence_length, hidden_dim)
        output = np.transpose(output, (1, 0, 2))
        return output, self.hidden_state

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the layer.

        This method computes the gradients of the loss w.r.t. the layer's weights, biases, and input. 
        It propagates the gradient to the previous layer in the network.

        #### Args
            - `output_gradient` (`np.ndarray`): The gradient of the loss w.r.t. the next layer's output.

        #### Returns
            - `np.ndarray`: The gradient of the loss w.r.t. the layer's input.
        """
        if self.activation is not None:
            output_gradient = self.activation.backward(output_gradient)

        # Gradients for weights and bias
        self.dweights = np.dot(self.input.T, output_gradient)
        self.dbias = np.sum(output_gradient, axis=0, keepdims=True)
        self.gradients = self.dweights, self.dbias

        # Gradient w.r.t. the input, to be passed to the previous layer
        dinput = np.dot(output_gradient, self.weights.T)

        return dinput

    def update(self, learning_rate: float, gradients: list[np.ndarray]) -> None:
        """Updates the layer's parameters (weights and biases) using the computed gradients.

        This method applies gradient descent to adjust the weights and biases of the layer, 
        minimizing the loss function during training.

        #### Args
            - `learning_rate` (`float`): The learning rate used to scale the gradient updates.
            - `gradients` (`list[np.ndarray]`): The list of computed gradients with which apply gradient descent.

        #### Returns
            - `None`: Updates the Layer's internal prameters and returns.
        """
        dweights, dbias = gradients
        self.weights -= learning_rate * dweights
        self.bias -= learning_rate * dbias
        self.trainable_params = self.weights, self.bias
        return None

if __name__ == "__main__":
    from layers.dense import Dense
    from data.encoders import OneHotEncoder

    batch_size = 2
    sequence_length = 3
    input_dim = 1
    vocab_length = 5
    hidden_dim = 32

    onehot = OneHotEncoder(vocab_length)
    # Generate a random sequence of `sequence_length` from a vocab of `vocab_length`.
    batch = []
    for _ in range(batch_size):
        input = np.random.choice(range(vocab_length), sequence_length, replace=True)
        input = onehot.encode(input.T)
        batch.append(input)
    batch = np.stack(batch)

    rnn = Recurrent(shape=(vocab_length, hidden_dim), weight_init='xavier', activation='tanh', name='rnn_1')
    dense = Dense(shape=(hidden_dim, vocab_length), weight_init='xavier', name='rnn_out')

    output, hidden = rnn.forward(batch)
    output = dense.forward(output)

    def check_len(a, n):
        """Check the length of a list."""
        assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'

    def check_shape(a, shape):
        """Check the shape of a tensor."""
        assert a.shape == shape, \
                f'tensor\'s shape {a.shape} != expected shape {shape}'

    check_len(output, batch_size)
    check_shape(output[0], (sequence_length, vocab_length))
    check_shape(rnn.hidden_state, (batch_size, hidden_dim))