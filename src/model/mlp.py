import numpy as np
from data.mnist_data import MNISTDatasetManager
from tqdm import tqdm, trange

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
        
        self.activation_fn = self.tanh_activation

        # Init hidden layers weights and bias
        self.weights, self.biases = [], []
        prev_dim = input_layer
        for dim in self.hidden_layer:
            self.weights.append(np.random.randn(prev_dim, dim) * 0.1)
            self.biases.append(np.zeros((1, dim)))
            prev_dim = dim

        # Init output layer and gradients w.r.t the weights and bias
        self.weights.append(np.random.randn(prev_dim, output_layer) * 0.1)
        self.biases.append(np.zeros((1, output_layer)))
        self.dW = [0] * self.nlayers
        self.db = [0] * self.nlayers
    
    def cross_entropy_loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Computes the cross-entropy loss between predicted probabilities and true labels.

            Args:
                y_hat (ndarray): Predicted probabilities from the forward pass.
                y (ndarray): The true target labels for each training sample.

            Returns:
                float: The cross-entropy loss averaged across all samples.
        """
        epsilon = 1e-8
        loss = -y * np.log(y_hat + epsilon) # Add epsilon for stability
        loss_batch = np.sum(loss) / y.shape[0]
        return loss_batch

    def sigmoid_activation(self, Z: np.ndarray, derivative: bool = False) -> np.ndarray:
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
    
    def tanh_activation(self, Z: np.ndarray, derivative: bool = False) -> np.ndarray:
        """Applies Tanh activation function to the input.

        The Sigmoid activation function maps input values to the range (-1, 1), useful for 
        dealing with negative values more effectively. 

        Args:
            Z (ndarray): Input array, typically a pre-activation value (logits) from a layer.
            derivative (bool, optional): Computes the derivative of the Sigmoid function instead of the activation itself.

        Returns:
            ndarray: Output array with the activation value or derivative for the layer.
        """
        tanh = np.divide(np.exp(Z) - np.exp(-Z), np.exp(Z) + np.exp(-Z)) # shortcut: np.tanh(Z)
        if derivative:
            return 1. - tanh**2
        return tanh

    def relu_activation(self, Z: np.ndarray, derivative: bool = False) -> np.ndarray:
        """Applies ReLU activation function to the input.

        A ReLU (Rectified Linear Unit) activation function is linear in the positive dimension, but zero in the negative dimension. 

        Args:
            Z (ndarray): Input array, typically a pre-activation value (logits) from a layer.
            derivative (bool, optional): Computes the derivative of the Sigmoid function instead of the activation itself.

        Returns:
            ndarray: Output array with the activation value or derivative for the layer.
        """
        if derivative:
            return np.where(Z < 0, 0, 1.)
        return np.maximum(0, Z)

    def softmax_activation(self, Z: np.ndarray) -> np.ndarray:
        """Applies Softmax activation function to the input.
        
        The softmax function converts logits (raw scores) into a probability distribution, 
        ensuring that the output values are in the range [0, 1] and sum to 1 across each sample.

        Args:
            Z (ndarray): Input array (logits). Each row corresponds to the raw scores for a single sample across all classes.

        Returns:
            ndarray: Output array with softmax probabilities. Each row represents a valid probability distribution.
        """
        exp_x = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        y_hat = exp_x / np.sum(exp_x, axis=1, keepdims=True) # predicted probability for each class
        return y_hat

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
            # Sigmoid Activation Layer
            h = self.activation_fn(Z)
            self.logits.append(h)

        # Output Layer
        Z = np.dot(self.logits[-1], self.weights[-1]) + self.biases[-1]
        self.logits.append(Z)
        y_hat = self.softmax_activation(Z)
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

            self.dW[i] = np.dot(self.logits[i].T, dloss)
            self.db[i] = np.sum(dloss, axis=0, keepdims=False)

            dloss = np.dot(dloss, self.weights[i].T)
        
        # Update weights and biases
        for i in range(self.nlayers):
            self.weights[i] -= self.learning_rate * self.dW[i]
            self.biases[i] -= self.learning_rate * self.db[i]

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
                    loss = self.cross_entropy_loss(y_hat, y_batch)

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
                    val_loss = self.cross_entropy_loss(val_probabilities, datamanager.validation_data[1])
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