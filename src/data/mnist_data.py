import numpy as np
from typing import Literal
import struct
from array import array
from math import ceil
import matplotlib.pyplot as plt
from PIL import Image
import time

from data.encoders import OneHotEncoder, SmoothLabelEncoder, Encoder

ENCODERS = {
    'onehot': OneHotEncoder,
    'smoothlabel': SmoothLabelEncoder
}

np.random.seed(42) # For reproducibility

MAX_PIXEL = 255

class MNISTDatasetManager:
    def __init__(self, batch_size: int, encoder: str):
        """MNIST Dataset Manager.

        Args:
            batch_size (int): Number of train samples used per batch.
        """
        self.batch_size = batch_size
        self.encoder: Encoder = ENCODERS[encoder]()
        self.train_data = None
        self.test_data = None
        self.validation_data = None

        self.AUGMENTATION_FN = [
            self._generate_rotation,
            self._generate_translation,
            self._generate_scaling,
            self._generate_shear,
            self._generatee_noise
        ]

    def __iter__(self):
        """Generates iterable mini-batches of training data."""
        if self.train_data is None:
            raise ValueError('No training data available.')
        
        images, labels = self.train_data
        data_length = images.shape[0]
        self.total_batches = ceil(data_length / self.batch_size)

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

    def load_data(
            self,
            images_filepath: str,
            labels_filepath: str,
            type: str = Literal['train', 'test'],
            validation_len: int = None
        ):
        """Loads training or test datasets by combining images and labels. Additionally, 
        it splits a validation subset from the data.
        
        Args:
            images_filepath (str): Path to the file containing the raw image data.
            labels_filepath (str): Path to the file containing the raw label data.
            type (str): Specifies the dataset type. Must be one of:
                - `'train'`: Loads data into `self.train_data`.
                - `'test'`: Loads data into `self.test_data`.
            validation_len (int, optional): The number of samples to allocate for a validation set.
        """
        images = self.load_images(images_filepath)
        labels = self.load_labels(labels_filepath)
        match type:
            case 'train':
                if validation_len:
                    images, labels = self._split_validation(images, labels, validation_len)
                self.train_data = images, labels
            case 'test':
                self.test_data = images, labels
            case _:
                raise ValueError(f"Type must be 'train' or 'test'.")

        return images, labels

    def _split_validation(self, images, labels, validation_len: int):
        val_images = images[:validation_len]
        val_labels = labels[:validation_len]
        self.validation_data = (val_images, val_labels)

        images = images[validation_len:]
        labels = labels[validation_len:]
        return images, labels

    def prepdata(
            self,
            type: str = Literal['train', 'validation', 'test'],
            shuffle: bool = False,
            transpose: bool = False
        ):
        """Prepares and preprocesses the MNIST dataset for training or testing.

        Flattens and Normalizes image pixel values, optionally shuffles the data

        Args:
            type (str): Specifies the dataset type (one of 'train' or 'test')
            shuffle (bool, optional): Shuffles the dataset.

        Returns:
            tuple: A tuple 2D np.ndarray's (images, labels).
        """
        if type == 'train':
            if self.train_data is None:
                raise ValueError('No training data available.')
            images, labels = self.train_data
            labels = self.encoder.encode(labels)
        elif type == 'validation':
            if self.validation_data is None:
                raise ValueError('No validation data available.')
            images, labels = self.validation_data
            labels = self.encoder.encode(labels)
        else:
            if self.test_data is None:
                raise ValueError('No test data available.')
            images, labels = self.test_data

        # Prep Data
        num_samples, num_rows, num_cols = images.shape
        images = np.transpose(images, axes=(0,2,1)) if transpose else images  # Transpose (i.e. EMNIST)
        images = images.reshape(num_samples, num_rows * num_cols)   # Flatten 28x28 images into 784 units.

        # Normalize
        images = np.divide(images, MAX_PIXEL)

        # Shuffle data
        if shuffle:
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            images = images[indices]
            labels = labels[indices]

        if type == 'train':
            self.train_data = images, labels
        elif type == 'validation':
            self.validation_data = images, labels
        else:
            self.test_data = images, labels

        return images, labels

    def augment(self, config):
        """
        Note: Excessive or inappropriate augmentation can lead to unrealistic samples that confuse the model. 
            For MNIST:
            - Rotation: ±15° to ±30°.
            - Translation: ≤10% of the image dimensions.
            - Scaling: 0.9x to 1.1x.
            Affine transformation:
                pixel(x,y) -> pixel(a x + b y + c, d x + e y + f)
        """
        print(f"Data Augmentation Started...")
        print(f"Dataset size: {len(self.train_data[0])} samples")
        start_time = time.time()

        self.config = config
        images, labels = self.train_data
        num_sections = 1
        for i, cfg in enumerate(self.config):
            if self.config[cfg]:
                augmented_images = []
                for raw_img in self.train_data[0]:
                    img = Image.fromarray(raw_img)
                    augmented_img = self.AUGMENTATION_FN[i](img)
                    augmented_images.append(augmented_img)
                images = np.append(images, augmented_images, axis=0)
                labels = np.append(labels, self.train_data[1])

                num_sections += 1

        self.train_data = images, labels

        end_time = time.time()
        if num_sections > 1:
            print(f"Data Augmentation Completed!")
            print(f"Total samples after augmentation: {len(self.train_data[0])}")
            print(f"Time Taken: {end_time - start_time:.2f} seconds")
        else:
            print(f"Data was not augmentated.")
            return None

        return np.vsplit(self.train_data[0], num_sections), np.split(self.train_data[1], num_sections)
    
    def _sample_extreme(self, min, max) -> int:
        """Return a value closer to the extremes between min and max"""
        sample = np.random.beta(0.4, 0.4)
        return min + sample * (max - min)
        
    def _generate_rotation(self, img: Image.Image) -> np.ndarray:
        """Rotates image by a given angle."""
        min = self.config['rotation'][0]
        max = self.config['rotation'][1]
        angle = self._sample_extreme(min, max)
        img_rot = img.rotate(angle, resample=Image.BILINEAR)
        return np.array(img_rot, dtype=np.float64)

    def _generate_translation(self, img: Image.Image):
        """Translate image by tx pixels horizontally and ty pixels vertically. """
        min = self.config['translation'][0]
        max = self.config['translation'][1]
        c = self._sample_extreme(min, max)
        f = self._sample_extreme(min, max)
        # Translation matrix
        matrix = (1, 0, c, 0, 1, f)
        pix = img.transform(img.size, method=Image.AFFINE, data=matrix)
        return np.array(pix, dtype=np.float64)

    def _generate_scaling(self, img: Image.Image):
        """Scale image by a factor"""
        min = self.config['scale'][0]
        max = self.config['scale'][1]
        a = self._sample_extreme(min, max)
        e = self._sample_extreme(min, max)
        matrix = (a, 0, 0, 0, e, 0)
        pix = img.transform(img.size, method=Image.AFFINE, data=matrix)
        return np.array(pix, dtype=np.float64)

    def _generate_shear(self, img: Image.Image):
        """Shear image symmetrically"""
        min = self.config['shear'][0]
        max = self.config['shear'][1]
        b = self._sample_extreme(min, max) * self.config['shear'][2] # skew left-right. No skew when 0
        d = self._sample_extreme(min, max) * self.config['shear'][3] # skew up-down
        matrix = (1, b, 0, d, 1, 0)
        pix = img.transform(img.size, method=Image.AFFINE, data=matrix)
        return np.array(pix, dtype=np.float64)

    def _generatee_noise(self, img: Image.Image):
        """Add random noise to the image."""
        array = np.array(img)
        noise_level = self.config['noise']
        noise = np.random.normal(0, 255 * noise_level, array.shape).astype(np.int32)
        noisy_array = np.clip(array + noise, 0, 255).astype(np.uint8)
        pix = Image.fromarray(noisy_array)
        return np.array(pix, dtype=np.float64)

def plot_images(filepath: str, images: list[np.ndarray], titles: list[str], rows: int, cols: int, reshape: None | tuple[int, int], cmap: plt.cm = plt.cm.gray):
    """Helper function to plot a list of images with their relating titles"""
    plt.figure(figsize=(30,20))
    for i, (image, title) in enumerate(zip(images, titles)):
        image = image.reshape(reshape[0], reshape[1]) if reshape else image
        plt.subplot(rows, cols, i+1)
        plt.imshow(image, cmap=cmap)
        if (title != ''):
            plt.title(title, fontsize = 15)
    plt.savefig(filepath)

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
                char_index = ceil(pixel * (len(block_chars) - 1) / MAX_PIXEL)
                mapped_char = block_chars[char_index]
                #pixel = ' ' if pixel == 0 else f'{pixel:.1f}'
                print(mapped_char, end=' ')
            print()

if __name__ == "__main__":

    import os
    import random
    from experiments.config import get_cfg_defaults

    random.seed(42) # For reproducibility

    # Load default configuration
    config = get_cfg_defaults()['dataset']

    # Load MINST dataset
    mnist = MNISTDatasetManager(
        config['batch_size'],
        config['encoder']
    )
    mnist.load_data(
        config['train_images_filepath'],
        config['train_labels_filepath'],
        'train', 
        validation_len = 10000
    )
    mnist.load_data(
        config['test_images_filepath'],
        config['test_labels_filepath'],
        'test'
    )

    # Data Augmentation
    x_train_augmented, y_train_augmented = mnist.augment(config['augmentation'])

    # Data Prep
    mnist.prepdata('train', shuffle = config['shuffle'], transpose = config['transpose'])
    mnist.prepdata('validation', transpose = config['transpose'])
    mnist.prepdata('test', transpose = config['transpose'])

    # Show some random training and test images 
    images, titles = [], []

    NUM_IMAGES = 20
    NUM_COLS = 5
    rows = int(NUM_IMAGES/NUM_COLS) + 1

    augmentations = [cfg for cfg in config['augmentation']]
    augmentations.insert(0, 'normal')

    train_indexes = list(random.randint(1, mnist.train_data[0].shape[0]//len(augmentations)) for _ in range(0, NUM_IMAGES))

    for x_section, y_section, aug_name in zip(x_train_augmented, y_train_augmented, augmentations):
        images, titles = [], []
        for i in train_indexes:
            images.append(x_section[i])
            titles.append(f'Training image [{i}] = {y_section[i]}')

        plot_path = os.path.join(config['plot_filepath'], f'MNIST_train_{aug_name}.png')
        plot_images(plot_path, images, titles, rows, NUM_COLS, reshape=(28,28), cmap=plt.cm.spring)
        print_images(images, titles, reshape=(28,28), whitebg=False)