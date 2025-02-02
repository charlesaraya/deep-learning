import numpy as np
from typing import Literal

from layers.layer import Layer

class Conv(Layer):
    """Implements a convolutional layer for Convolutional Neural Networks (CNNs).

    A convolutional layer applies convolutional operations to input data, enabling the extraction of spatial features.
    """
    def __init__(
        self,
        shape: tuple,
        kernel_num: int,
        kernel_size: int = 3,
        weight_init: str = Literal['random', 'xavier', 'he'],
        stride: int = 1,
        padding: int = 0,
        **kwargs
    ):
        """Initialize the ConvLayer layer.

        It supports configurable kernel size, stride, padding, and weight initialization strategies.

        #### Args
            - 'shape' (`tuple`): The shape of the input data, specified as (batch_size, channels, input height, input width)).
            - `kernel_num` (`int`): The number of filters (kernels) used.
            - `kernel_size` (`int`): The size of the filter kernel.
            - `weight_init` (`str`): The weight initilization mode.
            - `stride` (`int`): The step size by which the kernel moves across the input feature map.
            - `padding` (`int`): The amount of zero-padding added to the input feature map's borders.
        """
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Extract indexes
        batch_size, input_channels, input_height, input_width = shape
        self.featmap_size = (input_height + 2*self.padding - self.kernel_size) // self.stride + 1

        super(Conv, self).__init__(shape, **kwargs)

        # Initiliaze kernel weights
        self.kernels = self.init_kernels(
            input_channels,
            self.kernel_num,
            self.kernel_size,
            weight_init
        )
        self.bias = np.zeros(kernel_num)
        self.set_trainable_params(self.kernels, self.bias)
        self.dkernels, self.dbias = self.init_gradients()

    def init_kernels(self, input_channels: int, kernel_num: int, kernel_size: int, weight_init: str):
        """Initiliase Kernel weights using a given strategy

        #### Args
            - `weight_init` (`str`): The strategy for initializing weights. Options include:
                - `'random'`: Initializes weights with small random values scaled by 0.01.
                - `'xavier'`: Uses the Xavier/Glorot uniform initialization, suitable for layers with sigmoid or tanh activations.
                - `'he'`: Uses He initialization, ideal for layers with ReLU or Leaky ReLU activations.
        """
        match weight_init:
            case 'random':
                return np.random.randn(kernel_num, input_channels, kernel_size, kernel_size) * 0.01
            case 'xavier':
                upper = np.sqrt(1.0 / kernel_size)
                lower = -upper
                return np.random.uniform(lower, upper, (kernel_num, input_channels, kernel_size, kernel_size))
            case 'he':
                return np.random.randn(kernel_num, input_channels, kernel_size, kernel_size) * np.sqrt(2 / kernel_size)

    def _pad_data(self, input_data: np.ndarray, padding: int) -> np.ndarray:
        """Adds zeroes to input data borders to account for edge information"""
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        return np.pad(input_data, pad_width, mode ="constant") if padding > 0 else input_data

    def _flip_kernel(self, kernel: np.ndarray):
        """Rotates the kernel 180 degrees."""
        return np.flip(np.flip(kernel, axis=0), axis=1)

    def forward(self, input_data: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Performs a forward pass through the layer.

        Applies convolution operations to the input data to produce the feature map tensor.

        #### Args
            - `input_data` (`np.ndarray`): The input data for the layer.

        #### Returns
            - `np.ndarray`: The output feature map of the layer result of the convolution operation.
        """
        # Apply padding
        self.input_data = self._pad_data(input_data, self.padding)

        # Init feature map
        batch_size = self.input_data.shape[0]
        self.featmap = np.zeros((batch_size, self.kernel_num, self.featmap_size, self.featmap_size))

        # Perform convolution
        for n in range(batch_size):
            for k, kernel in enumerate(self.kernels):
                # Slide kernel filter across the image (left-right & top-down)
                for i in range(self.featmap_size):
                    # Set kernel window row indeces
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    for j in range(self.featmap_size):
                        # Set kernel window col indeces
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # Extract input data region on which compute convolution operation
                        region = self.input_data[n, :, h_start:h_end, w_start:w_end]
                        self.featmap[n, k, i, j] = np.sum(region * kernel) + self.bias[k]
        return self.featmap

    def backward(self, output_gradient: np.ndarray):
        """Performs the backward pass through the layer.

        This method computes the gradients of the loss w.r.t. the kernel weights, bias, and input.

        #### Args
            - `output_gradient` (`np.ndarray`): The gradient of the loss w.r.t. the next layer's output.

        #### Returns
            - `np.ndarray`: The gradient of the loss w.r.t. the layer's input.
        """
        # Init gradients
        dinput = np.zeros_like(self.input_data)
        self.dkernels = np.zeros_like(self.kernels)

        batch_size = self.input_data.shape[0]
        # Gradient w.r.t. kernel weights and input
        for n in range(batch_size):
            for k, kernel in enumerate(self.kernels):
                for i in range(self.featmap_size):
                    # Set kernel window row indeces
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    for j in range(self.featmap_size):
                        # Set kernel window col indeces
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        # Extract input data region on which compute gradient
                        region = self.input_data[n, :, h_start:h_end, w_start:w_end]

                        # Gradient w.r.t. the kernel weights
                        self.dkernels[k] += region * output_gradient[n, k, i, j]

                        # Gradient w.r.t. the input
                        dinput[n, :, h_start:h_end, w_start:w_end] += self._flip_kernel(kernel) * output_gradient[n, k, i, j]

        # Gradient w.r.t. biases
        self.dbias = np.sum(output_gradient, axis=(0, 2, 3))

        # Remove padding
        if self.padding > 0:
            dinput = dinput[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dinput

    def update(self, learning_rate: float) -> None:
        """Performs the update pass through the layer.

        This method applies gradient descent to adjust the kernel weights and biases of the layer, 
        minimizing the loss function during training.

        #### Args
            - `learning_rate` (`float`): The learning rate used to scale the gradient updates. 

        #### Returns
            - `None`: Updates the Layer's internal prameters and returns.
        """
        self.kernels -= learning_rate * self.dkernels
        self.bias -= learning_rate * self.dbias
        return None

if __name__ == "__main__":
    import time
    from math import ceil
    from tqdm import tqdm

    np.random.seed(42) # Ensure reproducibility

    # Data
    training_samples_test = 10
    training_samples = 50000
    batch_size = 32
    training_iterations_test = ceil(training_samples_test/batch_size)
    total_training_iterations = ceil(training_samples/batch_size)
    image_size = 10
    channels = 3

    # Convolution hyperparams
    input_data = np.random.randint(0, 255, (batch_size, channels, image_size, image_size))/255
    kernel_num = 3
    kernel_size = 3
    stride = 1
    padding = 1
    featmap_size = (image_size + 2*padding - kernel_size) // stride + 1

    # Pooling hyperparams
    pool_size = 2
    pool_stride = 2
    pool_padding = 0
    pool_mode = 'avg'
    poolmap_size = (featmap_size + 2*pool_padding - pool_size) // pool_stride + 1

    cov1 = Conv(
        (batch_size, channels, image_size, image_size),
        kernel_num = kernel_num,
        kernel_size = kernel_size,
        weight_init = 'random',
        stride = stride,
        padding = padding
    )

    # Test Summary
    print(f"\n{"─" * 15}Convolution Layer hyperparams{"─" * 15}")
    print(f"Input:\t\t({batch_size}, {image_size}, {image_size})\t\t(Batch size, Input Height, Input Width)") # M rows x N cols
    print(f"Kernels:\t({batch_size}, {kernel_num}, {kernel_size}, {kernel_size})\t\t(Batch size, Kernels, Kernel Height, Kernel Width)")
    print(f"Convolution:\t({kernel_size}, {stride}, {padding})\t\t(Kernel size, Stride, Padding)")
    print(f"Feature Map:\t({batch_size}, {kernel_num}, {featmap_size}, {featmap_size})\t(Batch size, Kernels, Feature Map Height, Feature Map Width)")

    print(f"{"─" * 45}")
    print(f"Params: {kernel_num * kernel_size**2 + 1:,} (Kernels * Kernel Size + Bias)")
    print(f"Complexity Est.: {kernel_num * featmap_size**2 * kernel_size**2:,} operations / sample." +
          "\tO(Kernels * Feature Map Size * Kernel Size)")
    print(f"{"─" * 45}")

    # Test Forward Pass
    print(f"\nTest ConvLayer forward pass ({training_samples_test} samples):")
    start_time = time.time()
    for i in tqdm(range(training_iterations_test)):
        output = cov1.forward(input_data)
    end_time = time.time()
    print(f"Forward pass completed.")

    # Calculate test time
    test_time = end_time - start_time
    formated_time = time.strftime("%H:%M:%S", time.gmtime(test_time))
    print(f"Test Training Time: {formated_time} seconds.")
    time_factor = ceil(training_samples/training_samples_test)
    formated_time = time.strftime("%H:%M:%S", time.gmtime(test_time * time_factor))
    print(f"Total Training Time ({training_samples} samples): {formated_time} seconds.\n")