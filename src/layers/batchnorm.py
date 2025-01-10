import numpy as np

from layers.layer import Layer

EPSILON = 1e-8

class BatchNorm(Layer):
    """Batch Normalization (BatchNorm) for deep learning models.
    
    BatchNorm normalizes the input data within each mini-batch during training and adjusts the distribution of activations.
    """
    def __init__(self, dim: int, momentum: float = 0.95):
        """Initializes the BatchNorm layer.

        Args:
            dim (int): Dimension of the layer.
            momentum (float, optional): Determines how much the current mini-batch contributes to the running averages.
        """
        super(BatchNorm, self).__init__(dim)
        self.momentum = momentum

        self.gamma = np.ones((1, dim))
        self.beta = np.zeros((1, dim))

        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))

    def __repr__(self):
        return f"BatchNorm(gamma={self.gamma.shape}, beta={self.beta.shape}, momentum={self.momentum})"

    def forward(self, Z: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Forward pass: normalizes the logits and applies scaling and shifting.

        During training, it computes the batch statistics and updates running statistics.
        During inference, it uses pre-computed running statistics.

        Args:
            Z (ndarray): Input logits to be batch-normalized.
            is_training (bool, optional): Indicates when the model is training.

        Returns:
            ndarray: The normalized and scaled output.
        """
        self.Z = Z

        if is_training:
            # Calculate batch logits' mean and variance
            self.mean = np.mean(self.Z, axis=0)
            self.var = np.var(self.Z, axis=0)
            # Normalize logits
            self.z_hat = (self.Z - self.mean) / np.sqrt(self.var + EPSILON)
            # Scale and shift
            self.out = self.gamma * self.z_hat  + self.beta
            # Update running statistics. Higher momentum makes the layer less sensitive to the current mini batch.
            self.running_mean = self.running_mean * self.momentum + self.mean * (1 - self.momentum)
            self.running_var = self.running_var * self.momentum + self.var * (1 - self.momentum)
        else: # while testing, we normalize the data using the pre-computed mean and variance
            z_norm = (self.Z - self.running_mean) / np.sqrt(self.running_var + EPSILON)
            self.out = self.gamma * z_norm + self.beta

        return self.out

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        """Backward pass: computes the gradients with respect to the inputs, gamma, and beta.

        Args:
            dloss (ndarray): The gradient of the loss with respect to the activations.

        Returns:
            ndarray: The gradient of the loss with respect to the input logits.
        """
        # Input: dloss (gradient of loss w.r.t. BN output)
        batch_size = dloss.shape[0]

        dout = batch_size * dloss

        # Gradients w.r.t. gamma and beta
        self.dgamma = np.sum(self.z_hat * dout, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        # break normalization formula into intermediate vars
        z_mu = self.Z - self.mean
        inv_sd = 1. / np.sqrt(self.var + EPSILON)

        # Gradients w.r.t. normalized logits z_hat
        dz_hat = dloss * self.gamma
        # Gradients w.r.t. variance
        dvar = np.sum((dz_hat * z_mu * (-0.5) * (inv_sd) ** 3), axis=0)
        # Gradients w.r.t. mean
        dmu = (np.sum((dz_hat * -inv_sd), axis=0)) + (dvar * (-2.0 / batch_size) * np.sum(z_mu, axis=0))

        # Gradients w.r.t. Z
        dloss1 = dz_hat * inv_sd
        dloss2 = dvar * (2.0 / batch_size) * z_mu
        dloss3 = (1.0 / batch_size) * dmu
        dloss = dloss1 + dloss2 + dloss3 # final partial derivatives, 
        return dloss