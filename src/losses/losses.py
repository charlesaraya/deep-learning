import numpy as np
from abc import ABC, abstractmethod

from losses.loss import Loss

class CrossEntropyLoss(Loss):
    """Implements the Cross-Entropy loss function.

    The Cross-Entropy loss measures the difference between the predicted probability 
    distribution and the true distribution, making it suitable for classification tasks.

    This loss is commonly used with models producing probabilistic outputs, such as 
    those with a softmax activation function in the output layer.

    #### Note
        - Suitable for multi-class and binary classification problems.
        - Requires the predicted probabilities to sum to 1 (e.g., softmax outputs).
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Computes the cross-entropy loss between predicted probabilities and true labels.

        #### Args
            - `y_hat` (`np.ndarray`): The redicted probabilities from the forward pass.
            - `y` (`np.ndarray`): The true target labels for each training sample.

        #### Returns
            - `float`: The cross-entropy loss averaged across all samples.
        """
        epsilon = 1e-8 # Add epsilon for stability
        loss = -np.sum(y * np.log(y_hat + epsilon), axis=1) # sample loss
        loss_batch = np.sum(loss) / y.shape[0] # batch average loss
        return loss_batch

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes the gradient w.r.t the loss.

        #### Args
            - `y_hat` (`np.ndarray`): The redicted probabilities from the forward pass.
            - `y` (`np.ndarray`): The true target labels for each training sample.

        #### Returns
            - `np.ndarray`: The gradient w.r.t the loss.
        """
        grad = (y_hat  - y) / y.shape[0]
        return grad

class MeanSquaredError(Loss):
    """Implements the Mean Squared Error (MSE) loss function.

    The MSE loss measures the average squared difference between predicted and actual values,
    making it suitable for regression task where the goal is to predict continuous outputs.

    #### Note
        - Suitable for regression problems such as predicting prices, temperatures, or other 
          continuous values.
        - Not suitable for classification tasks, as it does not handle probabilistic outputs 
          or categorical labels effectively.
    """
    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Computes the mean squared error between predicted probabilities and true labels.

        #### Args
            - `y_hat` (`np.ndarray`): The predicted probabilities from the forward pass.
            - `y` (`np.ndarray`): The true target labels for each training sample.

        #### Returns
            - `float`: The mean squared error averaged across all samples.
        """
        error = np.sum((y - y_hat)**2, axis=1) / y.shape[1]
        error_batch = np.sum(error) / y.shape[0]
        return error_batch

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes the gradient w.r.t the loss.

        #### Args
            - `y_hat` (`np.ndarray`): The redicted probabilities from the forward pass.

        #### Returns
            - `np.ndarray`: The gradient w.r.t the loss.
        """
        N, C = y.shape
        grad = (2 / (N * C)) * y_hat * ((y_hat  - y)  - np.sum((y_hat  - y) * y_hat, axis=1, keepdims=True))
        return grad

LOSS_FN = {
    'cross-entropy-loss': CrossEntropyLoss,
    'mean-squared-error': MeanSquaredError,
    None: None
}