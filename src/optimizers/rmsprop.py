import numpy as np

from optimizers.optimizer import Optimizer

class RMSProp(Optimizer):
    """Implements RMSProp optimizer.
    """
    def __init__(self, decay_rate: float = 0.9, **kwargs):
        """Initializes the RMSProp optimizer.

        #### Args
            - decay_rate (float):
        """
        super(RMSProp, self).__init__(**kwargs)
        self.decay_rate = decay_rate
        self.model_velocities = []

    def init_params(self, trainable_parameters: list[np.ndarray]) -> None:
        """Initializes velocity terms for each trainable parameter.
        #### Args
            - trainable_parameters (list[np.ndarray]): A list of parameter tensors from the model's layers.
        """
        layer_velocities = [None] * len(trainable_parameters)
        for idx, param in enumerate(trainable_parameters):
            layer_velocities[idx] = np.zeros_like(param)
        self.model_velocities.append(layer_velocities)
        return None

    def update(self, layer_id: int, gradients: list[np.ndarray]):
        """Updates gradient velocities using RMSProp.

        This method modifies the velocity of the gradients for the given layer 
        and returns the updated velocity values.

        #### Args
            - layer_id (int): The index of the layer whose parameters are being updated.
            - gradients (np.ndarray): The list of computed gradients for the layer parameters.

        Returns:
            list: The updated velocity values for the layer.

        Raises:
            RuntimeError: If the optimizer has not been initialized.
            IndexError: If the provided `layer_id` is out of range.
        """
        if not self.model_velocities:
            raise RuntimeError("Optimizer is not initiliazed.")
        elif layer_id < 0 or layer_id >= len(self.model_velocities):
            raise IndexError(f"Index is out of range. Tried to access the {layer_id} from an array of length {len(self.model_velocities)}.")

        computed_gradient = []
        for idx, gradient in enumerate(gradients):
            velocity = self.model_velocities[layer_id][idx]
            # Exponential Moving Average of Squared Gradients
            self.model_velocities[layer_id][idx] = self.decay_rate * velocity + (1 - self.decay_rate) * gradient**2
            # Parameter update rule: θ_t+1 = θ_t − (η / v_t + ϵ) * g
            # hence, computes the gradient g_t scaled down by sqrt(v_t)
            computed_gradient.append(gradient / np.sqrt(self.model_velocities[layer_id][idx] + 1e-8))

        return computed_gradient