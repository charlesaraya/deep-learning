import numpy as np

class MLP(object):
    """A simple implementation of a Multi-Layer Perceptron (MLP) for basic machine learning tasks."""

    def __init__(self, input_layer, hidden_layer, output_layer):
        """Initializes the MLP model with specified layer sizes and learning rate."""
        self.hidden_layer = hidden_layer
        self.nlayers = len(hidden_layer) + 1 # hidden layers + output layer
        
        self.weights = []
        self.biases = []

        # Hidden layers
        prev_dim = input_layer
        for dim in self.hidden_layer:
            self.weights.append(np.random.randn(prev_dim, dim) * 0.1)
            self.biases.append(np.zeros((1, dim)))
            prev_dim = dim

        # Output layer
        self.weights.append(np.random.randn(prev_dim, output_layer) * 0.1)
        self.biases.append(np.zeros((1, output_layer)))

        self.dW = [0] * self.nlayers
        self.db = [0] * self.nlayers

    def one_hot_encode(self, y_target):
        """Returns the one-hot encoding of each label class"""
        nlabels = max(y_target) + 1
        y_encoded  = []
        for yi in y_target:
            y_encoded.append(np.zeros(nlabels))
            y_encoded[-1][yi] = 1
        return np.asarray(y_encoded)

    def forward(self, X):
        """Forward pass through the network."""
        self.logits = []

        # Input layer
        self.logits.append(X)

        # Hidden Layers
        for i in range(len(self.hidden_layer)):
            # Linear Transform (Weighted Sum) Layer
            Z = np.dot(self.logits[-1], self.weights[i]) + self.biases[i]
            # Sigmoid Activation Layer
            h = 1.0 / (1 + np.exp(-Z))
            self.logits.append(h)

        # Output Layer
        Z = np.dot(self.logits[-1], self.weights[-1]) + self.biases[-1]
        self.logits.append(Z)
        # Apply Softmax Activation to Output Layer
        exp_x = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        y_hat = exp_x / np.sum(exp_x, axis=1, keepdims=True) # predicted probability for each class
        return y_hat

    def backward(self, y_hat, y):
        """Backward pass to compute gradients and update weights."""

        # Gradient of the loss w.r.t. predictions
        grad_loss = y_hat  - y
        dinput = grad_loss / y.shape[0]

        for i in reversed(range(self.nlayers)):
            if i < len(self.hidden_layer): # Skip output layer
                dinput = dinput * (self.logits[i+1] * (1 - self.logits[i+1]))

            self.dW[i] = np.dot(self.logits[i].T, dinput)
            self.db[i] = np.sum(dinput, axis=0, keepdims=False)

            dinput = np.dot(dinput, self.weights[i].T)
        
        # Update weights and biases
        for i in range(self.nlayers):
            self.weights[i] -= self.learning_rate * self.dW[i]
            self.biases[i] -= self.learning_rate * self.db[i]

    def train(self, train_data, epochs, learning_rate):
        """Train the MLP on provided training data."""
        training_accuracies, training_losses = [], []
        self.learning_rate = learning_rate

        X = train_data[0]
        y = self.one_hot_encode(train_data[1])
    
        for epoch in range(epochs):
            # Forward Pass
            y_hat = self.forward(X)

            # Calculate error in prediction using Cross-Entropy Loss function
            loss = np.mean(np.sum(-np.log(y_hat) * y, axis=1))
            
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