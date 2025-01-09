import numpy as np
from data.mnist_data import MNISTDatasetManager
from tqdm import tqdm, trange

from layers.activations import ACTIVATION_FN
import layers.losses as loss_fn
from layers.batchnorm import BatchNorm

np.random.seed(42)

class MLP:
    """A simple implementation of a Multi-Layer Perceptron (MLP) for basic machine learning tasks."""

    def __init__(self, input_layer: int, hidden_layer: list[int], output_layer: int):
        """Initializes the MLP model with specified layer sizes.

        Args:
            input_layer (int): Number of neurons in the input layer. Corresponds to the number of features in the input data.
            hidden_layer (list[int]): Number of neurons in the hidden layer. The list defines the architecture of the hidden layers.
            output_layer (int): Number of neurons in the output layer. Corresponds to the number of output classes for classification.
        """
        self.hidden_layer = hidden_layer
        self.nlayers = len(hidden_layer) + 1 # hidden layers + output layer
        
        self.activation_fn = ACTIVATION_FN['tanh']

        # Init hidden layers
        self.weights, self.biases = [], []
        self.batch_norm_layers: list[BatchNorm] = []

        prev_dim = input_layer
        for dim in self.hidden_layer:
            self.weights.append(np.random.randn(prev_dim, dim) * 0.1)
            self.biases.append(np.zeros((1, dim)))
            self.batch_norm_layers.append(BatchNorm(dim))
            prev_dim = dim

        # Init output layer and gradients w.r.t the weights and bias
        self.weights.append(np.random.randn(prev_dim, output_layer) * 0.1)
        self.biases.append(np.zeros((1, output_layer)))
        self.dW = [0] * self.nlayers
        self.db = [0] * self.nlayers

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Performs the forward pass through the network.

        Args:
            X (ndarray): Input data. Each row corresponds to a sample and each column corresponds to a feature.

        Returns:
            ndarray: Output of the network. Each row corresponds to the predicted values for each sample, and each column corresponds to a target label.
        """
        self.logits = []

        # Input layer
        self.logits.append(X)

        # Hidden Layers
        for i in range(len(self.hidden_layer)):
            # Linear Transform (Weighted Sum) Layer
            Z = np.dot(self.logits[-1], self.weights[i]) + self.biases[i]
            # BatchNorm Layer
            Z = self.batch_norm_layers[i].forward(Z)
            # Sigmoid Activation Layer
            h = self.activation_fn(Z)
            self.logits.append(h)

        # Output Layer
        Z = np.dot(self.logits[-1], self.weights[-1]) + self.biases[-1]
        self.logits.append(Z)
        y_hat = ACTIVATION_FN['softmax'](Z)
        return y_hat

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> None:
        """Performs the backward pass through the network.
        
        Computing the gradients of the loss w.r.t. the model parameters, and updates the weights and biases.

        Args:
            y_hat (ndarray): Predicted probabilities from the forward pass.
            y (ndarray): The true target labels for each training sample.
        """

        # Gradient of the loss w.r.t. predictions
        dloss = (y_hat  - y) / y.shape[0]

        for i in reversed(range(self.nlayers)):
            if i < len(self.hidden_layer): # Skip output layer
                dloss = dloss * self.activation_fn(self.logits[i+1], derivative=True)
                dloss = self.batch_norm_layers[i].backward(dloss)

            self.dW[i] = np.dot(self.logits[i].T, dloss)
            self.db[i] = np.sum(dloss, axis=0, keepdims=False)

            dloss = np.dot(dloss, self.weights[i].T)
        
        # Update weights and biases
        for i in range(self.nlayers):
            self.weights[i] -= self.learning_rate * self.dW[i]
            self.biases[i] -= self.learning_rate * self.db[i]
        # Update BatchNorm's learning parameters gamma and beta
        for i, bn in enumerate(self.batch_norm_layers):
            bn.gamma -= bn.dgamma * self.learning_rate
            bn.beta -= bn.dbeta * self.learning_rate

    def train(self, datamanager: MNISTDatasetManager, epochs: int, learning_rate: float) -> dict:
        """Trains the MLP on the training data.
        
        Performs forward and backward passes at a given learning rate, and over a number of epochs.

        Args:
            datamanager (MNISTDatasetManager): A DataManager class containing the training and validation data, as well as an iterator for mini-batch.
            epochs (int): The number of times the model will iterate over the entire training dataset.
            learning_rate (float): The learning rate used for gradient descent to update the model parameters.

        Returns:
            dict: Dictionary containing the following:
            - 'weights' (list[ndarray]): Final weights of the model.
            - 'bias' (list[ndarray]): Final biases of the model.
            - 'training_accuracies' (list): Training accuracy values recorded at each epoch.
            - 'training_losses' (list): Training loss values recorded at each epoch.
        """
        training_accuracies, training_losses, validation_accuracies, validation_losses = [], [], [], []
        self.learning_rate = learning_rate

        with trange(epochs) as t:
            for epoch in t:
                t.set_description(f"Epoch {epoch}") # Monitor epoch progress in terminal
                batch_accuracies, batch_losses = [], []
                for X_batch, y_batch in datamanager:
                    # Forward Pass
                    y_hat = self.forward(X_batch)

                    # Calculate error in prediction using Cross-Entropy Loss function
                    loss = loss_fn.cross_entropy_loss(y_hat, y_batch)

                    # Backpropagation Pass: Calculate Gradients, Weights & Bias
                    self.backward(y_hat, y_batch)

                    # Monitor batch metrics
                    predictions = np.argmax(y_hat, axis=1)
                    accuracy = np.mean(predictions == np.argmax(y_batch, axis=1))
                    batch_accuracies.append(accuracy)
                    batch_losses.append(loss)
                # Monitor epoch metrics
                epoch_loss = batch_losses[-1]
                epoch_accuracy = batch_accuracies[-1]
                training_accuracies.append(epoch_accuracy)
                training_losses.append(epoch_loss)

                # Validation
                if datamanager.validation_data:
                    # Accuracy
                    val_probabilities = self.forward(datamanager.validation_data[0])
                    val_predictions = np.argmax(val_probabilities, axis=1)
                    val_labels = np.argmax(datamanager.validation_data[1], axis=1)
                    val_accuracy = np.mean(val_predictions == val_labels)
                    validation_accuracies.append(val_accuracy)
                    # Loss
                    val_loss = loss_fn.cross_entropy_loss(val_probabilities, datamanager.validation_data[1])
                    validation_losses.append(val_loss)

                # Monitoring Metrics
                t.set_postfix(
                    tLoss = epoch_loss,
                    tAcc = epoch_accuracy*100,
                    vLoss = val_loss,
                    vAcc = val_accuracy*100
                )

        return {
            'weights': self.weights,
            'bias': self.biases,
            'training_accuracies': training_accuracies,
            'training_losses': training_losses,
            'validation_accuracies': validation_accuracies,
            'validation_losses': validation_losses
        }
    
if __name__ == "__main__":

    # Set file paths based on added MNIST Datasets
    config = {
        'train_images_filepath': './data/MNIST/train-images',
        'train_labels_filepath': './data/MNIST/train-labels',
        'test_images_filepath': './data/MNIST/test-images',
        'test_labels_filepath': './data/MNIST/test-labels',
        'metrics_filepath': './plots/metrics/',
    }

    # Load MINST dataset
    batch_size = 64
    mnist = MNISTDatasetManager(batch_size)

    mnist.load_data(
        config['train_images_filepath'],
        config['train_labels_filepath'],
        'train'
        )
    mnist.load_data(
        config['test_images_filepath'],
        config['test_labels_filepath'],
        'test'
        )

    train_data = mnist.prepdata('train', shuffle=True, validation_len=10000)
    test_data = mnist.prepdata('test')

    # Architecture
    input_layer = train_data[0].shape[1]
    hidden_layer = [64]
    output_layer = train_data[1].shape[1]
    # i.e "mlp_model[in-hl-hl-out]" 
    nn_arq = f'mlp_model[{input_layer}'
    nn_arq += ''.join(f'-{hl}' for hl in hidden_layer)
    nn_arq += f'-{output_layer}]'

    number_epochs = [10]
    learning_rate = 1e-3

    for epochs in number_epochs:

        # Setup NN
        mlp = MLP(input_layer, hidden_layer, output_layer)

        # Train
        output = mlp.train(mnist, epochs, learning_rate)

        # Inference
        test_probabilities = mlp.forward(test_data[0])
        test_predictions = np.argmax(test_probabilities, axis=1)

        # Accuracy
        test_accuracy = np.mean(test_predictions == test_data[1])

        # Results
        print(f"\n{nn_arq}, Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate} \
                \n{"─" * 15} Loss {"─" * 20} \
                \nTraining Loss:\t{output['training_losses'][-1]:.3} \
                \nValid Loss:\t{output['validation_losses'][-1]:.3} \
                \n{"─" * 15} Accuracies {"─" * 15} \
                \nTraining Acc.:\t{output['training_accuracies'][-1]:.3%} \
                \nValid Acc.:\t{output['validation_accuracies'][-1]:.3%} \
                \nTest Acc.:\t{test_accuracy:.3%}\n")