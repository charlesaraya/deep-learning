import numpy as np
from tqdm import tqdm, trange

from data.mnist_data import MNISTDatasetManager
from layers.layer import Layer
from layers.denselayer import DenseLayer
import layers.losses as loss_fn
from layers.activations import ACTIVATION_FN

class BaseModel:

    def __init__(self):
        self.layers: list[Layer] = []

    def add(self, layer: Layer):
        """Adds a layer to the model"""
        self.layers.append(layer)

    def __str__(self):
        self.name = f'model[{self.layers[0].shape[0]}'
        self.name += ''.join(f'-{layer.shape[1]}' for layer in self.layers)
        self.name += ']'
        return self.name
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Performs the forward pass through the network.

        Args:
            X (ndarray): Input data. Each row corresponds to a sample and each column corresponds to a feature.

        Returns:
            ndarray: Output of the network. Each row corresponds to the predicted values for each sample, and each column corresponds to a target label.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad: np.ndarray) -> None:
        """Performs the backward pass through the network.

        Computing the gradients of the loss w.r.t. the model parameters, and updates the weights and biases.

        Args:
            y_hat (ndarray): Predicted probabilities from the forward pass.
            y (ndarray): The true target labels for each training sample.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

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
                    # Calculate gradient w.r.t loss
                    grad = (y_hat  - y_batch) / y_batch.shape[0]
                    # Backpropagation Pass: Calculate Gradients, Weights & Bias
                    self.backward(grad)

                    # Gradient Descent: Update Weights and Biases
                    for layer in self.layers:
                        if isinstance(layer, DenseLayer):
                            layer.weights -= self.learning_rate * layer.dweights
                            layer.bias -= self.learning_rate * layer.dbias

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
                
        self.weights, self.biases = [(layer.weights, layer.bias) for layer in self.layers if isinstance(layer, DenseLayer)]

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

    number_epochs = [20]
    learning_rate = 1e-3

    for epochs in number_epochs:

        # Setup NN
        mlp = BaseModel()
        mlp.add(DenseLayer(input_layer, 64, activation='tanh'))
        mlp.add(DenseLayer(64, output_layer, activation='softmax'))

        # Train
        output = mlp.train(mnist, epochs, learning_rate)

        # Inference
        test_probabilities = mlp.forward(test_data[0])
        test_predictions = np.argmax(test_probabilities, axis=1)

        # Accuracy
        test_accuracy = np.mean(test_predictions == test_data[1])

        # Results
        print(f"\n{mlp.__str__()}, Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate} \
                \n{"─" * 15} Loss {"─" * 20} \
                \nTraining Loss:\t{output['training_losses'][-1]:.3} \
                \nValid Loss:\t{output['validation_losses'][-1]:.3} \
                \n{"─" * 15} Accuracies {"─" * 15} \
                \nTraining Acc.:\t{output['training_accuracies'][-1]:.3%} \
                \nValid Acc.:\t{output['validation_accuracies'][-1]:.3%} \
                \nTest Acc.:\t{test_accuracy:.3%}\n")