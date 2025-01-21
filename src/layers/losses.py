import numpy as np
from abc import ABC, abstractmethod

class Loss:
    def __init__(self):
        pass

    @abstractmethod
    def forward(y_hat: np.ndarray, y: np.ndarray) -> float:
        """Forward pass"""
        pass

    @abstractmethod
    def backward(y_hat: np.ndarray, y: np.ndarray) -> float:
        """Backward pass"""
        pass

class CrossEntropyLoss(Loss):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    
    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Computes the cross-entropy loss between predicted probabilities and true labels.

            Args:
                y_hat (ndarray): Predicted probabilities from the forward pass.
                y (ndarray): The true target labels for each training sample.

            Returns:
                float: The cross-entropy loss averaged across all samples.
        """
        epsilon = 1e-8 # Add epsilon for stability
        loss = -np.sum(y * np.log(y_hat + epsilon), axis=1) # sample loss
        loss_batch = np.sum(loss) / y.shape[0] # batch average loss
        return loss_batch

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Computes the gradient w.r.t the loss."""
        grad = (y_hat  - y) / y.shape[0]
        return grad
    

class MeanSquaredError(Loss):

    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Computes the mean squared error between predicted probabilities and true labels.

            Args:
                y_hat (ndarray): Predicted probabilities from the forward pass.
                y (ndarray): The true target labels for each training sample.

            Returns:
                float: The mean squared error averaged across all samples.
        """
        error = np.sum((y - y_hat)**2, axis=1) / y.shape[1]
        error_batch = np.sum(error) / y.shape[0]
        return error_batch

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Computes the gradient w.r.t the loss."""
        N, C = y.shape
        grad = (2 / (N * C)) * y_hat * ((y_hat  - y)  - np.sum((y_hat  - y) * y_hat, axis=1, keepdims=True))
        return grad

LOSS_FN = {
    'cross-entropy-loss': CrossEntropyLoss,
    'mean-squared-error': MeanSquaredError,
    None: None
}