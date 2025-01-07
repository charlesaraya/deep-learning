import numpy as np
import struct
from array import array
import random
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

np.random.seed(42) # For reproducibility

class MNISTDatasetManager(object):
    def __init__(self, batch_size: int):
        """MNIST Dataset Manager.

        Args:
            batch_size (int): Number of train samples used per batch.
        """
        self.batch_size = batch_size
        self.train_data = None
        self.test_data = None
        self.validation_data = None

    def __iter__(self):
        """Generates iterable mini-batches of training data."""
        if self.train_data is None:
            raise ValueError('No training data available.')
        
        images, labels = self.train_data
        data_length = images.shape[0]
        self.total_batches = -(data_length // -self.batch_size) # def ceildiv(a, b): return -(a // -b)

        data_indices = np.arange(data_length)
        for start_idx in range(0, len(images), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = data_indices[start_idx:end_idx]
            yield images[batch_indices], labels[batch_indices]

    def load_labels(self, filepath: str):
        """Loads labels from the specified file path in raw binary format.

        Reads and parses label data from a binary file, typically used for datasets 
        such as MNIST. The file's structure is validated using a magic number, and the labels 
        are extracted as integers.

        Args:
            filepath (str): Path to the file or directory containing the label raw data.

        Returns:
            np.ndarray: Array of labels, where each label is an integer (e.g., 0 to 9).

        Raises:
            ValueError: If the file's magic number does not match the expected value (2049).

        File Format:
            The label file format is as follows:
            - Offset 0000: Magic number (4 bytes; 3rd byte indicates data type, 4th byte indicates dimensions).
            - Offset 0004: Dataset size (number of labels).
            - Offset 0008+: Labels (unsigned bytes, one per label).
        References:
            - [Yann LeCun's MNIST Dataset Format](https://yann.lecun.com/exdb/mnist/)
        """
        with open(filepath, 'rb') as file:
            magic, size = struct.unpack('>II', file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = np.asarray(array('B', file.read())) # next bytes represent the labels values (0 to 9)
        return labels

    def load_images(self, filepath: str):
        """Loads images from the specified file path in raw binary format.

        Reads and parses image data from a binary file.

        Args:
            filepath (str): Path to the file or directory containing the image raw data.

        File Format:
            The image file format is as follows:
            - Offset 0000: Magic number (4 bytes; identifies the file type).
            - Offset 0004: Dataset size (number of images).
            - Offset 0008: Number of rows per image.
            - Offset 0012: Number of columns per image.
            - Offset 0016+: Pixel values (unsigned bytes, row-major order).
        """
        with open(filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack('>IIII', file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = array('B', file.read())

        shape = (size, rows, cols)
        images = np.zeros(shape)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i] = img

        return images

    def load_data(self, images_filepath: str, labels_filepath: str, type: str):
        """Loads training or test datasets by combining images and labels.
        
        Args:
            images_filepath (str): Path to the file containing the raw image data.
            labels_filepath (str): Path to the file containing the raw label data.
            type (str): Specifies the dataset type. Must be one of:
                - `'train'`: Loads data into `self.train_data`.
                - `'test'`: Loads data into `self.test_data`.
        """
        images = self.load_images(images_filepath)
        labels = self.load_labels(labels_filepath)
        match type:
            case 'train':
                self.train_data = images, labels
            case 'test':
                self.test_data = images, labels
            case _:
                raise ValueError(f"Type must be 'train' or 'test'.")

        return images, labels

    def one_hot_encode(self, labels: np.ndarray):
        """Encodes each target label class into its one-hot format.

        Args:
            labels (ndarray): Array of integers representing the target labels.
        """
        nlabels = max(labels) + 1
        labels_encoded  = []
        for label in labels:
            labels_encoded.append(np.zeros(nlabels))
            labels_encoded[-1][label] = 1
        return np.asarray(labels_encoded)

    def prepdata(self, type: str, shuffle: bool = False, validation_len: int = None):
        """Returns preprocessed the training and test datasets."""
        match type:
            case 'train':
                if self.train_data is None:
                    raise ValueError('No training data available.')
                images, labels = self.train_data
                labels = self.one_hot_encode(labels)
            case 'test':
                if self.test_data is None:
                    raise ValueError('No test data available.')
                images, labels = self.test_data

        # Prep Data
        data_length, N, M = images.shape
        images = images.reshape(data_length, N * M)   # Flatten 28x28 images into 784 units.

        # Normalioze
        images = np.divide(images, 255.0)

        # Shuffle data
        if shuffle:
            data_indices = np.arange(data_length)
            np.random.shuffle(data_indices)
            images = images[data_indices]
            labels = labels[data_indices]

        # Prep validation Data
        if validation_len:
            indices = np.arange(data_length)
            # Split validation set from the training set
            val_images = images[indices[:validation_len]]
            val_labels = labels[indices[:validation_len]]
            self.validation_data = (val_images, val_labels)

            images = images[indices[validation_len:]]
            labels = labels[indices[validation_len:]]

        match type:
            case 'train':
                self.train_data = images, labels
                return self.train_data
            case 'test':
                self.test_data = images, labels
                return self.test_data

def plot_images(images: list[np.ndarray], titles: list[str], rows: int, cols: int, reshape: None | tuple[int, int], cmap: plt.cm = plt.cm.gray):
    """Helper function to plot a list of images with their relating titles"""
    plt.figure(figsize=(30,20))
    for i, (image, title) in enumerate(zip(images, titles)):
        image = image.reshape(reshape[0], reshape[1]) if reshape else image
        plt.subplot(rows, cols, i+1)
        plt.imshow(image, cmap=cmap)
        if (title != ''):
            plt.title(title, fontsize = 15)
    plt.savefig(f'./plots/mnist_image_labels.png')

def print_images(images: list[np.ndarray], title_texts: list[str], reshape: None | tuple[int, int], whitebg: bool = True):
    """Helper function to print a list of images with their relating titles"""
    # Block characters in increasing order of density
    block_chars = [' ', '░', '▒', '▓', '█']
    block_chars if whitebg else block_chars.reverse()
    for i, image in enumerate(images):
        print(title_texts[i])
        image = image.reshape(reshape[0], reshape[1]) if reshape else image
        for row in image:
            for pixel in row:
                char_index = (pixel * (len(block_chars) - 1) + 0.25).astype(int) 
                mapped_char = block_chars[char_index]
                #pixel = ' ' if pixel == 0 else f'{pixel:.1f}'
                print(mapped_char, end=' ')
            print()

if __name__ == "__main__":

    # Set file paths based on added MNIST Datasets
    config = {
        'train_images_filepath': './data/MNIST/train-images',
        'train_labels_filepath': './data/MNIST/train-labels',
        'test_images_filepath': './data/MNIST/test-images',
        'test_labels_filepath': './data/MNIST/test-labels'
    }

    # Load MINST dataset
    mnist = MNISTDatasetManager(batch_size = 64)

    mnist.load_data(
        config['train_images_filepath'],
        config['train_labels_filepath'],
        'train'
        )
    mnist.load_data(
        config['test_images_filepath'],
        config['test_labels_filepath'],
        'test'
        )
    
    (x_train, y_train) = mnist.prepdata('train', shuffle = True, validation_len = 10000)
    (x_test, y_test) = mnist.prepdata('test')

    # Show some random training and test images 
    images, titles = [], []

    train_indexes = (random.randint(1, len(x_train)) for _ in range(0, 20))
    test_indexes = (random.randint(1, len(x_test)) for _ in range(0, 10))

    for r in train_indexes:
        images.append(x_train[r])
        titles.append('Training image [' + str(r) + '] = ' + str(y_train[r]))

    for r in test_indexes:
        images.append(x_test[r])
        titles.append('Test image [' + str(r) + '] = ' + str(y_test[r]))

    cols = 5
    rows = int(len(images)/cols) + 1
    plot_images(images, titles, rows, cols, reshape=(28,28), cmap=plt.cm.spring)
    print_images(images, titles, reshape=(28,28), whitebg=False)