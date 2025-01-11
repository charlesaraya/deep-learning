import numpy as np

from layers.layer import Layer

class Dropout (Layer):
    """Dropout layer for regularizing neural networks.

    Dropout randomly deactivates a fraction of the input units during training to reduce overfitting. 
    During inference, the activations are scaled to maintain consistency with the training phase.
    """
    def __init__(self, rate: float = 0.5):
        """Initializes the Dropout layer.
        
        Args:
            rate (float): The dropout rate (the probability of dropping an neuron. i.e.)
        """
        super(Dropout, self).__init__()
        if not (0 < rate < 1):
            raise ValueError("Rate must be between 0 and 1.")
        self.rate = rate

    def forward(self, Z: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Forward pass across the Dropout layer.

        During training, it applies a mask to the activations that sets a random 
        fractions of the inputs tu 0, and scaling down the remaining activations.
        During inference, dropout is turned off, and all neurons are used.

        Args:
            Z (ndarray): The input activations to the dropout layer.
            is_training (bool, optional): Indicates when the model is training.

        Returns:
            ndarray: The output after applying dropout (or unchanged during inference).
        """
        if is_training:
            self.mask = (np.random.uniform(size=Z.shape) > self.rate).astype(float)
            return Z * self.mask / (1 - self.rate)

        else:
            return Z

    def backward(self, dloss) -> np.ndarray:
        """Backward pass through the Dropout layer.

        During backpropagation, the gradient is scaled by the same dropout mask used in the 
        forward pass.

        Args:
            dloss (ndarray): The gradient of the loss with respect to the layer's output.

        Returns:
            ndarray: The gradient of the loss with respect to the input logits.
        """
        return dloss * self.mask / (1 - self.rate)