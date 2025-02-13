import numpy as np

from optimizers.optimizer import Optimizer

class SGD(Optimizer):
    """Implements Stochastic Gradient Descent (SGD) with Momentum.

    This optimizer updates model parameters using the gradient of the loss function 
    while incorporating momentum to accelerate learning and dampen oscillations.
    """
    def __init__(
        self,
        momentum: float = 0.0,
        clip_gradient_value = None,
        **kwargs,
    ):
        """Initializes the SGD with Momentum optimizer.

        #### Args
            - momentum (float): A value between 0 and 1 that controls the contribution 
                of the previous gradient updates.
            - **kwargs: Additional arguments passed to the parent `Optimizer` class.
        """
        super(SGD, self).__init__(
            clip_gradient_value = clip_gradient_value,
            **kwargs,
        )
        self.momentum = momentum
        self.model_velocities = []

    def init_params(self, trainable_parameters: list[np.ndarray]) -> None:
        """Initializes velocity terms for each trainable parameter.

        This method prepares the velocity matrices required for momentum-based 
        updates by setting them to zero.

        #### Args
            - trainable_parameters (list[np.ndarray]): A list of parameter tensors from the model's layers.
        """
        layer_velocities = [None] * len(trainable_parameters)
        for idx, param in enumerate(trainable_parameters):
            layer_velocities[idx] = np.zeros_like(param)
        self.model_velocities.append(layer_velocities)
        return None

    def update(self, layer_id: int, gradients: list[np.ndarray]):
        """Updates model parameters using SGD with momentum.

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
        for idx, gradient in enumerate(gradients):
            velocity = self.model_velocities[layer_id][idx]
            gradient = self.momentum * velocity + (1 - self.momentum) * gradient
            gradient = self.clip_gradients(gradient)
            self.model_velocities[layer_id][idx] = gradient
        return self.model_velocities[layer_id]

if __name__ == "__main__":

    sgd = SGD(name="my_sgd_optimizer")

    weights = np.random.randn(4,10) * 0.01
    bias = np.zeros((1,10))

    learning_rate = 0.01

    dgrads = sgd.update(learning_rate, [weights, bias], momentum=0.5)

    print(dgrads)