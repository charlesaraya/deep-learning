import numpy as np

def cross_entropy_loss(y_hat: np.ndarray, y: np.ndarray) -> float:
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

LOSS_FN = {
    'cross-entropy': cross_entropy_loss,
    None: None
}