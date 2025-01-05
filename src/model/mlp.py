import numpy as np

np.random.seed(42)

class MLP(object):
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

    def one_hot_encode(self, y_target: np.ndarray) -> np.ndarray:
        """Encodes each target label class into its one-hot format.

        Args:
            y_target (ndarray): Array of integers representing the target labels.

        Returns:
            ndarray: Array of float arrays representing the one-hot encoded vector of the target labels.
        """
        nlabels = max(y_target) + 1
        y_encoded  = []
        for yi in y_target:
            y_encoded.append(np.zeros(nlabels))
            y_encoded[-1][yi] = 1
        return np.asarray(y_encoded)
    
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
    
    def mini_batch_data(self, X: np.ndarray, y: np.ndarray, batch_size: int):
        """Generates mini-batches of data for training.

        This function splits the input data into smaller subsets (mini-batches) of the specified size and yields one mini-batch at a time.

        Args:
            X (ndarray): Input data. Each row corresponds to a sample and each column corresponds to a feature.
            y (ndarray): The true target labels for each training sample.
            batch_size (int): The size of each mini-batch.

        Yields:
            tuple: A tuple (X_batch, y_batch) where:
                - X_batch (ndarray): Mini-batch of input data. The last batch could be smaller if the data size is not divisible by the batch size.
                - y_batch (ndarray): Mini-batch of target labels.
        """
        data_indices = np.arange(X.shape[0])
        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = start_idx + batch_size
            batch_indices = data_indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]

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

    def train(self, train_data: np.ndarray, epochs: int, learning_rate: float, batch_size: int) -> dict:
        """Trains the MLP on the training data.
        
        Performs forward and backward passes at a given learning rate, and over a number of epochs.

        Args:
            train_data (tuple): A tuple containing the training data, where:
                - X (ndarray): The input data.
                - y (ndarray): The true target labels that exist in the classification task.
            epochs (int): The number of times the model will iterate over the entire training dataset.
            learning_rate (float): The learning rate used for gradient descent to update the model parameters.

        Returns:
            dict: Dictionary containing the following:
            - 'weights' (list[ndarray]): Final weights of the model.
            - 'bias' (list[ndarray]): Final biases of the model.
            - 'training_accuracies' (list): Training accuracy values recorded at each epoch.
            - 'training_losses' (list): Training loss values recorded at each epoch.
        """
        training_accuracies, training_losses = [], []
        self.learning_rate = learning_rate

        X = train_data[0]
        y = self.one_hot_encode(train_data[1])
    
        for epoch in range(epochs):
            batch_accuracies, batch_losses = [], []
            for X_batch, y_batch in self.mini_batch_data(X, y, batch_size):
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
            training_accuracies.append(np.mean(batch_accuracies))
            training_losses.append(np.mean(batch_losses))

        return {
            'weights': self.weights,
            'bias': self.biases,
            'training_accuracies': training_accuracies,
            'training_losses': training_losses
        }

if __name__ == '__main__':
    import pandas as pd

    # Data Prep
    df = pd.read_csv('data/Iris/iris.csv', header=None)
    df[5] = pd.Categorical(df[4]).codes
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Sample test and train data
    test_ratio = 0.2
    test_data = df.sample(frac=test_ratio, random_state=42)
    train_data = df.drop(test_data.index)
    # Train Dataset
    x_inputs_train = train_data.iloc[:, 0:4].values
    y_target_output_train = np.array(train_data[5])
    train_data = (x_inputs_train, y_target_output_train)
    # Test Dataset
    x_inputs_test = test_data.iloc[:,0:4].values
    y_target_output_test = np.array(test_data[5])
    test_data = (x_inputs_test, y_target_output_test)

    # Hyperparameters
    input_layer = 4
    hidden_layer = [64]
    output_layer = 3
    learning_rate = 0.01
    epochs = 20
    batch_size = 8

    # NN Setup
    p = MLP(input_layer, hidden_layer, output_layer)

    # Training
    output = p.train((x_inputs_train, y_target_output_train), epochs, learning_rate, batch_size)

    # Inference
    test_probabilities = p.forward(test_data[0])
    test_predictions = np.argmax(test_probabilities, axis=1)

    # Accuracy
    test_accuracy = np.mean(test_predictions == test_data[1])

    # i.e "mlp_model[in-hl-hl-out]" 
    nn_arq = f'mlp_model[{input_layer}'
    nn_arq += ''.join(f'-{hl}' for hl in hidden_layer)
    nn_arq += f'-{output_layer}]'

    # Results
    print(f"\n{nn_arq}, epochs: {epochs}, batch size: {batch_size}, learning rate: {learning_rate} \
            \nTraining Loss:\t{output['training_losses'][-1]:.3} \
            \nTraining Acc.:\t{output['training_accuracies'][-1]:.3%} \
            \nTest Acc.:\t{test_accuracy:.3%}\n")