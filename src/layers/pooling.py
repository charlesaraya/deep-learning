import numpy as np
from typing import Literal

from layers.layer import Layer

class Pooling(Layer):
    """A pooling layer for Convolutional Neural Networks (CNNs) that performs dimensionality 
    reduction (downsampling) by applying a pooling operation (e.g., max or average) to input feature maps,
    thus retaining the most important or representative features.
    It enhances efficiency and robustness without contributing to the model's capacity to learn relationships in the data.
    """
    def __init__(
            self,
            input_shape: tuple,
            window_size: int,
            stride: int,
            padding: int,
            mode: str = 'max'
        ):
        """Initiliaze Pool Layer

        Args:
            input_size (int): The size of the input feature map data, specified as height x width.
            window_size (int): The size of the pooling window.
            stride (int): The step size for sliding the pooling window across the input.
            mode (str): The pooling mode, either 'max' for max pooling or 'avg' for average pooling.
        
        Methods:
            forward(input):
                Applies the pooling operation to the input feature maps during the forward pass.
            
            backward(grad_output):
                Computes the gradient of the loss with respect to the input during the backward pass.
                This is typically used in backpropagation to propagate gradients through the layer.
        """
        self.pool_size = window_size
        self.stride = stride
        self.padding = padding

        # Extract indexes
        batch_size, self.kernel_num, input_height, input_width = input_shape
        self.poolmap_size = (input_height + 2*self.padding - self.pool_size) // self.stride + 1

        input_size = batch_size * self.kernel_num * input_height * input_width
        output_size = batch_size * self.kernel_num * self.poolmap_size**2
        super(Pooling, self).__init__(input_size, output_size)

        POOLING_FN = {
            'max': self._max_pooling,
            'avg': self._avg_pooling
        }
        if mode not in POOLING_FN:
            raise ValueError(f"Unsupported ppoling operation '{mode}'")
        self.pooling_operation = POOLING_FN[mode]

    def _max_pooling(self, X: np.ndarray, mode: str = Literal['forward', 'backward']) -> float:
        if mode == 'forward':
            return np.max(X)
        elif mode == 'backward':
            max = np.max(X)
            gradient_map = (X == max).astype(int)
            return gradient_map

    def _avg_pooling(self, X: np.ndarray, mode: str = Literal['forward', 'backward']) -> float:
        if mode == 'forward':
            return np.mean(X)
        elif mode == 'backward':
            gradient_map = np.ones(X.shape) / X.size
            return gradient_map

    def forward(self, input_data: np.ndarray, is_training: bool = True):
        """Applies the pooling operation to the feature map tensor."""
        # Apply padding to input
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        self.input_data = np.pad(input_data, pad_width, mode ="constant") if self.padding > 0 else input_data

        # Init pool map
        batch_size = input_data.shape[0]
        self.poolmap = np.zeros((batch_size, self.kernel_num, self.poolmap_size, self.poolmap_size))

        # Perform pooling across the batch
        for n in range(batch_size):
            # across each kernel
            for k in range(self.kernel_num):
                # slide pool window across the feature map: left-right & top-down
                for i in range(self.poolmap_size):
                    h_start = i * self.stride
                    h_end = h_start + self.pool_size
                    # calculate feature map
                    for j in range(self.poolmap_size):
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        # extract region to which compute pooling operation
                        region = self.input_data[n, k, h_start:h_end, w_start:w_end]
                        self.poolmap[n, k, i, j] = self.pooling_operation(region, mode='forward')
        return self.poolmap

    def backward(self, output_gradient: np.ndarray):
        """Computes the gradient of the loss with respect to the input."""
        # Init gradients
        doutput = np.zeros_like(self.input_data)

        batch_size = self.input_data.shape[0]
        # Perform pooling across the batch
        for n in range(batch_size):
            # across each kernel
            for k in range(self.kernel_num):
                # slide pool window across the feature map: left-right & top-down
                for i in range(self.poolmap_size):
                    h_start = i * self.stride
                    h_end = h_start + self.pool_size
                    # calculate feature map
                    for j in range(self.poolmap_size):
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        # extract region to which compute pooling operation
                        region = self.input_data[n, k, h_start:h_end, w_start:w_end]
                        dregion = self.pooling_operation(region, mode='backward') * output_gradient[n, k, i, j]
                        doutput[n, k, h_start:h_end, w_start:w_end] = dregion
        return doutput

if __name__ == "__main__":
    import time
    from math import ceil
    from tqdm import tqdm

    from layers.activations import ReLU

    # Data
    training_samples_test = 100
    training_samples = 50000
    batch_size = 64
    training_iterations_test = ceil(training_samples_test/batch_size)
    total_training_iterations = ceil(training_samples/batch_size)
    image_dim = 28

    # Convolution hyperparams
    kernel_size = 3
    feat_map_stride = 2
    image_padding = 1
    featmap_size = (image_dim + 2*image_padding - kernel_size) // feat_map_stride + 1
    kernel_num = 5
    relu = ReLU()
    activation = relu.forward(np.random.normal(0, 1, (batch_size, kernel_num, featmap_size, featmap_size)))

    # Pool hyperparams
    pool_size = 2
    stride = 2
    padding = 0
    poolmap_size = (featmap_size + 2*padding - pool_size) // stride + 1

    pool = Pooling(
        input_shape = (batch_size, kernel_num, featmap_size, featmap_size),
        window_size = pool_size,
        stride = stride,
        padding = padding,
        mode = 'avg'
    )

    # Test Summary
    print(f"\n{"─" * 15}Pool Layer hyperparams{"─" * 15}")
    print(f"Feature Map:\t({batch_size}, {kernel_num}, {featmap_size}, {featmap_size})\t(Batch Size, Kernels, Feature Map Shape)")
    print(f"Pooling:\t({pool_size}, {stride}, {padding})\t\t(Pool Window size, Stride, Padding)")
    print(f"Pool Map:\t({batch_size}, {kernel_num}, {poolmap_size}, {poolmap_size})\t(Batch Size, Kernels, Pool Map Shape)\n")

    print(f"{"─" * 45}")
    print(f"Params: {kernel_num * poolmap_size**2:,} (Kernels * Pool Map Size)")
    print(f"Complexity Est.: {kernel_num * poolmap_size**2 * pool_size**2:,} operations / sample." +
          "\tO(Kernels * Pool Map Size * Pool Window Size)")
    print(f"{"─" * 45}")

    # Test Forward Pass
    print(f"\nTest PoolLayer forward pass ({training_samples_test} samples):")
    start_time = time.time()
    for i in tqdm(range(training_iterations_test)):
        output = pool.forward(activation)
    end_time = time.time()
    print(f"Forward pass completed.")

    # Calculate test time
    test_time = end_time - start_time
    formated_time = time.strftime("%H:%M:%S", time.gmtime(test_time))
    print(f"Test Training Time: {formated_time} seconds.")
    time_factor = ceil(training_samples/training_samples_test)
    formated_time = time.strftime("%H:%M:%S", time.gmtime(test_time * time_factor))
    print(f"Total Training Time ({training_samples} samples): {formated_time} seconds.\n")

    # Test Backward Pass
    grad = np.random.normal(0, 1, (batch_size, kernel_num, featmap_size, featmap_size))
    print(f"\nTest PoolLayer backward pass ({training_samples_test} samples):")
    start_time = time.time()
    for i in tqdm(range(training_iterations_test)):
        doutput = pool.backward(grad)
    end_time = time.time()
    print(f"Forward pass completed.")