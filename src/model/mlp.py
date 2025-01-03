import numpy as np

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

    def one_hot_encode(self, y_target):
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
    
    def cross_entropy_loss(self, y_hat, y):
        """Computes the cross-entropy loss between predicted probabilities and true labels.

            Args:
                y_hat (ndarray): Predicted probabilities from the forward pass.
                y (ndarray): The true target labels for each training sample.

            Returns:
                float: The cross-entropy loss averaged across all samples.
        """
        epsilon = 1e-8
        loss = -np.mean(np.sum(y * np.log(y_hat + epsilon), axis=1)) # Add epsilon for stability
        return loss

    def sigmoid_activation(self, Z, derivative: bool = False):
        """Applies Sigmoid activation function to the input."""
        if derivative:
            return Z * (1 - Z)
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
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
            h = self.sigmoid_activation(Z)
            self.logits.append(h)

        # Output Layer
        Z = np.dot(self.logits[-1], self.weights[-1]) + self.biases[-1]
        self.logits.append(Z)
        # Apply Softmax Activation to Output Layer
        exp_x = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        y_hat = exp_x / np.sum(exp_x, axis=1, keepdims=True) # predicted probability for each class
        return y_hat

    def backward(self, y_hat, y):
        """Performs the backward pass through the network.
        
        Computing the gradients of the loss w.r.t. the model parameters, and updates the weights and biases.

        Args:
            y_hat (ndarray): Predicted probabilities from the forward pass.
            y (ndarray): The true target labels for each training sample.
        """

        # Gradient of the loss w.r.t. predictions
        grad_loss = y_hat  - y
        dinput = grad_loss / y.shape[0]

        for i in reversed(range(self.nlayers)):
            if i < len(self.hidden_layer): # Skip output layer
                dinput = dinput * self.sigmoid_activation(self.logits[i+1], derivative=True)

            self.dW[i] = np.dot(self.logits[i].T, dinput)
            self.db[i] = np.sum(dinput, axis=0, keepdims=False)

            dinput = np.dot(dinput, self.weights[i].T)
        
        # Update weights and biases
        for i in range(self.nlayers):
            self.weights[i] -= self.learning_rate * self.dW[i]
            self.biases[i] -= self.learning_rate * self.db[i]

    def train(self, train_data, epochs, learning_rate):
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
            # Forward Pass
            y_hat = self.forward(X)

            # Calculate error in prediction using Cross-Entropy Loss function
            loss = self.cross_entropy_loss(y_hat, y)
            
            # Backpropagation Pass: Calculate Gradients, Weights & Bias
            self.backward(y_hat, y)

            # Monitor Accuracy and Loss
            predictions = np.argmax(y_hat, axis=1)
            accuracy = np.mean(predictions == np.argmax(y, axis=1))
            training_accuracies.append(accuracy)
            training_losses.append(loss)
        
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
    # Sample test and train data
    test_ratio = 0.2
    test_data = df.sample(frac=test_ratio, random_state=8000)
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
    hidden_layer = [128]
    output_layer = 3
    learning_rate = 0.01
    epochs = 2000

    # NN Setup
    p = MLP(input_layer, hidden_layer, output_layer)

    # Training
    output = p.train((x_inputs_train, y_target_output_train), epochs, learning_rate)

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
    print(f"\n{nn_arq}, epochs: {epochs}, learning rate: {learning_rate} \
            \nTraining Loss:\t{output['training_losses'][-1]:.3} \
            \nTraining Acc.:\t{output['training_accuracies'][-1]:.3%} \
            \nTest Acc.:\t{test_accuracy:.3%}\n")