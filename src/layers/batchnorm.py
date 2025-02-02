import numpy as np

from layers.layer import Layer

EPSILON = 1e-8

class BatchNorm(Layer):
    """Implements batch normalization for deep learning models.

    Batch normalization normalizes the input of each mini-batch to have a mean of 0 and a variance of 1, 
    improving training stability and accelerating convergence. 

    It also introduces learnable parameters for scaling (gamma) and shifting (beta) the normalized output.
    """
    def __init__(self, shape: tuple, momentum: float = 0.95, **kwargs):
        """Initializes the BatchNorm layer.

        #### Args
            - `momentum` (`float`, optional): Momentum for the running mean and variance updates (default is 0.95). 
            Larger values make the running statistics adapt more slowly to new data, while smaller values allow faster adaptation.
        """
        super(BatchNorm, self).__init__(shape, **kwargs)

        self.momentum = momentum
        self.total_params = 2 * shape[0]

        self.axis_op = None
        self.is_initiliazed = False

    def init_layer(self, input_data: np.ndarray) -> None:
        self.batch_size = input_data.shape[0] 
        nchannels = input_data.shape[1]

        # Calculate the number of dimensions to expand (first 2 are fixed)
        ndim = input_data.ndim
        expand_dims = ndim - 2

        # Dynamically expand dimensions
        self.shape = (1, nchannels) + (1,) * expand_dims

        # Init parameters
        self.gamma = np.ones(self.shape)
        self.beta = np.zeros(self.shape)
        self.set_trainable_params(self.gamma, self.beta)
        self.dgamma, self.dbeta = self.init_gradients()

        self.running_mean = np.zeros(self.shape)
        self.running_var = np.ones(self.shape)

        # Extract axes to which calculate mean and var
        if expand_dims > 0:
            axes = [i for i in range(ndim)]
            data_ndim = ndim // 2
            self.axis_op = (0, *axes[-data_ndim:])
        # special case: input_data with shape (batch size, input_size) hence compute ops only along axis 0
        elif expand_dims == 0:
            self.axis_op = 0

        self.is_initiliazed = True
        return None

    def __repr__(self):
        return f"BatchNorm({self.shape}, gamma={self.gamma.shape}, beta={self.beta.shape}, momentum={self.momentum})"

    def forward(self, input_data: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Performs a forward pass through the layer.

        Normalizes the logits and applies scaling and shifting. 
        During training, it computes the batch statistics and updates running statistics.
        During inference, it uses the pre-computed running statistics.

        #### Args
            - `Z` (`np.ndarray`): Input logits to be batch-normalized.
            - `is_training` (`bool`, optional): Indicates when the model is training.

        #### Returns
            - `np.ndarray`: The normalized and scaled output.
        """
        self.X = input_data
        # lazily initiliaze the layer (feat: makes it agnostic to X shape)
        if not self.is_initiliazed:
            self.init_layer(self.X)

        if is_training:
            # Calculate batch logits' mean and variance
            self.mean = np.mean(self.X, axis=self.axis_op, keepdims=True)
            self.var = np.var(self.X, axis=self.axis_op, keepdims=True)
            # Normalize logits
            self.Z = (self.X - self.mean) / np.sqrt(self.var + EPSILON)
            # Scale and shift
            self.out = self.gamma * self.Z  + self.beta
            # Update running statistics. Higher momentum makes the layer less sensitive to the current mini batch.
            self.running_mean = self.running_mean * self.momentum + self.mean * (1 - self.momentum)
            self.running_var = self.running_var * self.momentum + self.var * (1 - self.momentum)
        else: # while testing, we normalize the data using the pre-computed mean and variance
            z_norm = (self.X - self.running_mean) / np.sqrt(self.running_var + EPSILON)
            self.out = self.gamma * z_norm + self.beta

        return self.out

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        """Performs the backward pass through the layer.

        This method computes the gradients w.r.t. thelayer's learning parameters, gamma, and beta, and w.r.t. the inputs.
        It propagates the gradient to the previous layer in the network.

        #### Args
            - `dloss` (`np.ndarray`): The gradient of the loss w.r.t. the  next layer's output, typically the activation layer.

        #### Returns
            - `np.ndarray`: The gradient of the loss w.r.t. the layer's input.
        """
        # Input: dloss (gradient of loss w.r.t. BN output)
        dout = self.batch_size * dloss

        # Gradients w.r.t. gamma and beta
        self.dgamma = np.sum(self.Z * dout, axis=self.axis_op, keepdims=True)
        self.dbeta = np.sum(dout, axis=self.axis_op, keepdims=True)

        # break normalization formula into intermediate vars
        z_mu = self.X - self.mean
        inv_sd = 1. / np.sqrt(self.var + EPSILON)

        # Gradients w.r.t. normalized logits Z
        dz_hat = dloss * self.gamma
        # Gradients w.r.t. variance
        dvar = np.sum((dz_hat * z_mu * (-0.5) * (inv_sd) ** 3), axis=0)
        # Gradients w.r.t. mean
        dmu = (np.sum((dz_hat * -inv_sd), axis=0)) + (dvar * (-2.0 / self.batch_size) * np.sum(z_mu, axis=0))

        # Gradients w.r.t. Z
        dloss1 = dz_hat * inv_sd
        dloss2 = dvar * (2.0 / self.batch_size) * z_mu
        dloss3 = (1.0 / self.batch_size) * dmu
        dloss = dloss1 + dloss2 + dloss3 # final partial derivatives, 
        return dloss

    def update(self, learning_rate: float) -> None:
        """Updates the layer's learning parameters (gamma and beta) using the computed gradients.

        This method applies gradient descent to adjust the learning parameters of the layer, 
        minimizing the loss function during training.

        #### Args
            - `learning_rate` (`float`): The learning rate used to scale the gradient updates. 

        #### Returns
            - `None`: Updates the Layer's internal prameters and returns.
        """
        self.gamma -= learning_rate * self.dgamma
        self.beta -= learning_rate * self.dbeta
        return None