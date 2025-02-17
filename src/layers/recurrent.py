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
        sequence_2_sequence = False,
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

        input_dim, hidden_dim, output_dim = shape
        # Init weights
        self.weights_x = self.init_weight((input_dim, hidden_dim), weight_init)
        self.weights_h = self.init_weight((hidden_dim, hidden_dim), weight_init)
        self.weights_y = self.init_weight((hidden_dim, output_dim), weight_init)

        # Init biases
        self.bias_h = np.zeros((1, hidden_dim))
        self.bias_y = np.zeros((1, output_dim))

        # Init hidden state
        self.hidden_state = None

        self.set_trainable_params(
            self.weights_x,
            self.weights_h,
            self.weights_y,
            self.bias_h,
            self.bias_y,
        )
        # Init gradients
        self.dweights_x, self.dweights_h, self.dweights_output, self.dbias_h, self.dbias_output = self.init_gradients()

        # Set activation function
        self.activation: Layer = ACTIVATIONS[activation]

        # Control forward pass output
        self.sequence_2_sequence = sequence_2_sequence

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
        # Shape of inputs: (batch_size, sequence_length, input_dim)
        self.input = input_data
        batch_size, sequence_length, _ = self.input.shape

        # Init hidden state. Shape: (batch_size, num_hiddens)
        hidden_dim = self.shape[1]
        """ if self.hidden_state is None:
            self.hidden_state = np.zeros((batch_size, hidden_dim)) """
        hidden_state_prev = np.zeros((batch_size, hidden_dim))

        outputs = []
        self.hidden_states = []
        for t_step in range(sequence_length):
            # Recurrent layer pass
            input_t = self.input[:, t_step, :] # Shape: (batch_size, input_dim)
            input_x = np.dot(input_t, self.weights_x)

            # Update hidden state
            hidden_state = np.dot(hidden_state_prev, self.weights_h) + self.bias_h

            # Activation
            hidden_state = self.activation.forward(hidden_state + input_x)
            self.hidden_states.append(hidden_state)

            # Output layer pass
            output_x = np.dot(hidden_state, self.weights_y) + self.bias_y
            outputs.append(output_x)

            hidden_state_prev = hidden_state.copy()

        outputs = np.stack(outputs, axis=0)
        outputs = np.transpose(outputs, axes=(1, 0, 2))
        self.hidden_states = np.stack(self.hidden_states, axis=0)
        self.hidden_states = np.transpose(self.hidden_states, axes=(1, 0, 2))

        if not self.sequence_2_sequence:
            outputs = outputs[:, -1, np.newaxis]
        return outputs

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the layer.

        This method computes the gradients of the loss w.r.t. the layer's weights, biases, and input. 
        It propagates the gradient to the previous layer in the network.

        #### Args
            - `output_gradient` (`np.ndarray`): The gradient of the loss w.r.t. the next layer's output.

        #### Returns
            - `np.ndarray`: The gradient of the loss w.r.t. the layer's input.
        """
        # Shape of inputs: (batch_size, sequence_length, input_dim)
        _, sequence_length, _ = self.input.shape
        output_gradient = np.transpose(output_gradient, axes=(1, 0, 2))
        self.hidden_states = np.transpose(self.hidden_states, axes=(1, 0, 2))

        dhidden_state_next = np.zeros_like(self.hidden_states[1])

        for t_step in reversed(range(sequence_length)):
            # Output layer pass: Gradient w.r.t output layer's weights, biases, & hidden state
            doutput_x = output_gradient[t_step]
            self.dweights_output += np.dot(self.hidden_states[t_step].T, doutput_x) # output_x = np.dot(hidden_state, self.weights_y) + self.bias_y
            self.dbias_output += np.sum(doutput_x, axis=0, keepdims=True)
            dhidden_output =  np.dot(doutput_x, self.weights_y.T)

            # Propagates gradient to hidden unit
            if t_step < sequence_length-1:
                dhidden_state = np.dot(dhidden_state_next, self.dweights_h.T) # hidden_state = np.dot(hidden_state_prev, self.weights_h) + self.bias_h

            # Activation: Pull gradient value across nonlinearity
            dhidden_state = self.activation.backward(dhidden_output + dhidden_state) # hidden_state = self.activation.forward(hidden_state + input_x)

            # Store to compute hidden unit gradient for previous sequence
            dhidden_state_next = dhidden_state.copy()

            # Recurrent layer pass: # Gradients w.r.t. weights, biases, and input
            if t_step > 0:  # If we're not at the very beginning
                self.dweights_h += np.dot(self.hidden_states[t_step-1].T, dhidden_state) # hidden_state = np.dot(hidden_state_prev, self.weights_h) + self.bias_h
                self.dbias_h += np.sum(dhidden_state, axis=0, keepdims=True)
            self.dweights_x = np.dot(self.input[:,t_step,:].T, dhidden_state)

        self.gradients = self.dweights_x, self.dweights_h, self.dweights_output, self.dbias_h, self.dbias_output

        return None

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
        dweights_x, dweights_h, dweights_y, dbias_h, dbias_y = gradients

        self.weights_x -= learning_rate * dweights_x
        self.weights_h -= learning_rate * dweights_h
        self.bias_h -= learning_rate * dbias_h
        self.weights_y -= learning_rate * dweights_y
        self.bias_y -= learning_rate * dbias_y

        self.trainable_params = self.weights_x, self.weights_h, self.weights_y, self.bias_h, self.bias_h
        return None

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    import math
    import pandas as pd

    from model.model import Model
    from data.datamanager import DatasetManager
    from losses.losses import MeanSquaredError
    from optimizers.schedulers import WarmUpScheduler, StepDecayScheduler
    from optimizers.sgd import SGD

    np.random.seed(0) # Reproducibility
    # Load the dataset
    batch_size = 10
    datamanager = DatasetManager(
        batch_size = batch_size,
        sequence_length = 7,
        sliding_window = True,
    )
    datamanager.load_data(
        filepath = './data/time_series/weather/clean_weather.csv',
        features = [1, 2, 3],
        target = [4],
        fillna = True,
        train_ratio = 0.9,
    )
    datamanager.prepdata()

    epochs = 10
    learning_rate = 1e-2
    learning_rate_start = 1e-6
    steps_per_epoch = math.ceil(datamanager.train_data[0].shape[0] / batch_size)
    steps_total = steps_per_epoch * epochs

    # Setup NN
    rnn = Model(name="rnn")

    # Build model by adding lñayers sequentially
    rnn.add(Recurrent((3, 4, 1), weight_init='xavier', activation='tanh'))

    rnn.compile(
        optimizer = SGD(clip_gradient_value=5),
        loss = MeanSquaredError(),
    )

    basemodel = StepDecayScheduler(learning_rate, step_size=steps_per_epoch, decay_factor=0.90)
    scheduler = WarmUpScheduler(basemodel, learning_rate_start, learning_rate, steps_per_epoch*2)

    # Train
    output = rnn.train(
        datamanager,
        scheduler,
        epochs,
    )

    # Inference
    test_probabilities = rnn.forward(datamanager.test_data[0], is_training=False)
    test_predictions = np.argmax(test_probabilities, axis=1)

    # Accuracy
    test_accuracy = np.mean(test_predictions == datamanager.test_data[1])

    # Results
    print(f"\n{rnn.__str__()}, Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate} \
            \n{"─" * 15} Loss {"─" * 20} \
            \nTraining Loss:\t{output['training_losses'][-1]:.3} \
            \nValid Loss:\t{output['validation_losses'][-1]:.3} \
            \n{"─" * 15} Accuracies {"─" * 15} \
            \nTraining Acc.:\t{output['training_accuracies'][-1]:.3%} \
            \nValid Acc.:\t{output['validation_accuracies'][-1]:.3%} \
            \nTest Acc.:\t{test_accuracy:.3%}\n")