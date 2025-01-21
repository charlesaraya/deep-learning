import numpy as np
from typing import Literal

from layers.layer import Layer

class ConvLayer(Layer):
    def __init__(
            self,
            input_shape: tuple[int, int],
            kernel_num: int,
            kernel_size: int = 3,
            weight_init: str = Literal['random', 'xavier', 'he'],
            stride: int = 1,
            padding: int = 0
        ):
        input_size = input_shape[0] * input_shape[1]
        output_dim = (input_shape[0] + (2 * padding) - kernel_size) // stride + 1
        self.output_shape = output_dim, output_dim
        super(ConvLayer, self).__init__(input_size, output_dim * output_dim)

        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Initiliaze kernels
        self.kernels = self.init_kernels(kernel_num, kernel_size, weight_init)
        self.bias = np.zeros(kernel_num)

    def init_kernels(self, kernel_num: int, kernel_size: int, weight_init: str):
        """Initiliase Kernel weights using a given strategy"""
        match weight_init:
            case 'random':
                return np.random.randn(kernel_num, kernel_size, kernel_size) * 0.01
            case 'xavier':
                upper = np.sqrt(1.0 / kernel_size)
                lower = -upper
                return np.random.uniform(lower, upper, (kernel_num, kernel_size, kernel_size))
            case 'he':
                return np.random.randn(kernel_num, kernel_size, kernel_size) * np.sqrt(2 / kernel_size)

    def _pad_data(self, X: np.ndarray, padding: int) -> np.ndarray:
        """Apply padding to data to account for edge information"""
        pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding))
        return np.pad(X, pad_width, mode ="constant") if padding > 0 else X

    def forward(self, input_data: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Forward pass"""
        batch_size, input_height, input_width = input_data.shape

        # Apply padding
        input_data = self._pad_data(input_data, self.padding)

        featmap_dim = (input_height + 2*self.padding - self.kernel_size) // self.stride + 1

        # Init feature map
        featmap = np.zeros((batch_size, self.kernel_num, featmap_dim, featmap_dim))

        # Perform convolution for each sample image in the batch
        for n in range(batch_size):
            # across each kernel
            for k, kernel in enumerate(self.kernels):
                # slide kernel filter across the image: left-right & top-down
                for i in range(featmap_dim):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    # calculate feature map
                    for j in range(featmap_dim):
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size

                        region = input_data[n, h_start:h_end, w_start:w_end]
                        featmap[n, k, i, j] = np.sum(region * kernel) + self.bias[k]
        return featmap

    def backward(self, output_gradient: np.ndarray):
        """Backward pass"""
        pass

if __name__ == "__main__":
    import time
    from math import ceil
    from tqdm import tqdm

    training_samples_test = 1000
    training_samples = 50000
    batch_size = 32
    training_iterations = ceil(training_samples_test/batch_size)
    image_dim = 28
    pixel_max = 255
    channel_num = 1
    data = np.random.randint(0, pixel_max, (batch_size, image_dim, image_dim))/pixel_max
    input_shape = data.shape[1], data.shape[2]
    kernel_num = 64
    kernel_size = 5
    stride = 2
    padding = 1
    featmap_dim = (image_dim + 2*padding - kernel_size) // stride + 1

    cnn = ConvLayer(input_shape, kernel_num, kernel_size, 'random', stride, padding)

    print(f"CNN hyperparams:")
    print(f"Input: ({batch_size}, {channel_num}, {image_dim}, {image_dim}) [Batch size, Channels, Img M, Img N]") # M rows x N cols
    print(f"Kernels: ({kernel_num}, {kernel_size}, {kernel_size}) [Kernels, K M, K N]")
    print(f"Feature Map: ({kernel_num}, {featmap_dim}, {featmap_dim}) [Kernels, FM M, FM N]\n")

    print(f"Num. Parameters: {kernel_num * channel_num * kernel_size**2 + 1}")
    print(f"Total Complexity: {training_iterations * batch_size * channel_num * featmap_dim**2 * kernel_num * kernel_size**2:,} operations.\n")

    print(f"\nTest ConvLayer forward pass on {training_samples_test} training samples:")
    start_time = time.time()
    for i in tqdm(range(training_iterations)):
        output = cnn.forward(data)
    end_time = time.time()
    print(f"Forward pass completed. Calculating total training time on {training_samples} samples.")
    test_time = end_time - start_time
    formated_time = time.strftime("%H:%M:%S", time.gmtime(test_time))
    print(f"Test Training Time: {formated_time} seconds.")
    time_factor = ceil(training_samples/training_samples_test)
    formated_time = time.strftime("%H:%M:%S", time.gmtime(test_time * time_factor))
    print(f"Estimated Total Training Time: {formated_time} seconds.\n")