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
MAX_VALIDATION_RATIO = 0.2  # 20% of the training set

class MNISTDatasetManager:
    def __init__(
            self,
            batch_size: int,
            encoder: str,
            nlabels: int,
            label_offset: int,
        ):
        """MNIST Dataset Manager.

        Args:
            batch_size (int): Number of train samples used per batch.
            encoder (Encoder): Encoder class that will encode label classes.
            nlabels (int): Number of class labels.
            label_offset (int): Offset of starting class label.
        """
        self.batch_size = batch_size
        self.encoder: Encoder = ENCODERS[encoder](nlabels, label_offset)

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
        images, labels = self._shuffle_data(images, labels) # Shuffle before each epoch reduce bias from the order of the data and speed up convergence.
        data_length = images.shape[0]
        self.total_batches = ceil(data_length / self.batch_size)

        data_indices = np.arange(data_length)
        for start_idx in range(0, len(images), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = data_indices[start_idx:end_idx]
            yield images[batch_indices], labels[batch_indices]

    def load_labels(self, filepath: str) -> np.ndarray:
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

    def load_images(self, filepath: str, channels: int = 1) -> np.ndarray:
        """Loads images from the specified file path in raw binary format.

        Reads and parses image data from a binary file.

        Args:
            filepath (str): Path to the file or directory containing the image raw data.
            channels (int, optional): The number of color channels in the image data (default = 0).
                - `1`: Grayscale images.
                - `3`: RGB images.

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

        shape = (size, channels, rows, cols)
        images = np.zeros(shape)
        for c in range(channels):
            for i in range(size):
                img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
                images[i, c] = img.reshape(rows, cols)

        return images

    def load_data(
            self,
            images_filepath: str,
            labels_filepath: str,
            channels: int = 1,
            type: str = Literal['train', 'test'],
        ):
        """Loads training or test datasets by combining images and labels. Additionally, 
        it splits a validation subset from the data.
        
        Args:
            images_filepath (str): Path to the file containing the raw image data.
            labels_filepath (str): Path to the file containing the raw label data.
            channels (int, optional): The number of color channels in the image data (default = 1).
                - `1`: Grayscale images.
                - `3`: RGB images.
            type (str): Specifies the dataset type. Must be one of:
                - `train`: Loads data into `self.train_data`.
                - `test`: Loads data into `self.test_data`.
        """
        images = self.load_images(images_filepath, channels)
        labels = self.load_labels(labels_filepath)
        match type:
            case 'train':
                self.train_data = images, labels
            case 'test':
                self.test_data = images, labels
            case _:
                raise ValueError(f"Type must be 'train' or 'test'.")

        return images, labels

    def _split_validation(self, validation_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Splits the validation set from the training set.

        Args:
            validation_ratio (float): The ratio of samples to include form the training set into the validation set (default = 0.1).

        Returns:
            tuple[np.ndarray, np.ndarray]: Returns the validation set.
        """
        if self.train_data is None:
            raise ValueError('No training data available.')

        if validation_ratio < 0 or validation_ratio > MAX_VALIDATION_RATIO:
            raise ValueError(f"Invalid value {validation_ratio} for `validation_ratio`. Expected value between 0 and `0.2`")

        images, labels = self.train_data
        val_len = ceil(len(images) * validation_ratio)
        self.validation_data = images[:val_len], labels[:val_len]
        self.train_data = images[val_len:], labels[val_len:]

        return self.validation_data

    def _shuffle_data(self, images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_samples = images.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        images = images[indices]
        labels = labels[indices]
        return images, labels

    def prepdata(
            self,
            validation_ratio: int = 0.1,
            shuffle: bool = False,
            flatten: bool = True,
            transpose: bool = False,
            debug_ratio: float = 1,
        ) -> None:
        """Prepares and preprocesses the training, validation, and test datasets.

        Flattens, Normalizes image pixel values, transposes the data, shuffles the data.

        Args:
            validation_ratio (float, optional): The percentage of samples to include in the validation set from the training set (default = 0.1).
                - `0`: No validation data will be created.
            shuffle (bool, optional): Shuffles the dataset (default = False).
            flatten (bool, optional): Flattens the dataset (default = True).
            transpose (bool, optional): Transposes the image data (default = False).
            debug_ratio (float, optional): Percentage of data to work with (default = 1).
                - `1`: Use 100% of the training set, hence no debugging.
                - `0.2`: Use 20% of the training set.

        Returns:
            tuple: A tuple 2D np.ndarray's (images, labels).
        """
        if self.train_data is None:
            raise ValueError('No training data available.')

        if self.test_data is None:
            raise ValueError('No test data available.')

        # Prep Data
        self.train_data = self.train_data[0], self.encoder.encode(self.train_data[1])

        for dataset_name in ["train_data", "test_data"]:
            # Access dataset dynamically
            images, labels = getattr(self, dataset_name)

            if shuffle:
                images, labels = self._shuffle_data(images, labels)

            if transpose:
                images = np.rot90(np.flip(images, axis=3), axes=(2, 3))

            if flatten:
                images = images.reshape(images.shape[0], -1)

            if dataset_name == "train_data":
                if debug_ratio > 0 and debug_ratio <= 1:
                    debug_len = ceil(len(images) * debug_ratio)
                    images = images[:debug_len]
                    labels = labels[:debug_len]
                else:
                    raise ValueError(f"Invalid debug factor {debug_ratio}. Should be kept between range (0, 1].")

            # Normalize
            images = np.divide(images, MAX_PIXEL)

            # Reassign dynamically back to the original dataset
            setattr(self, dataset_name, (images, labels))

        if validation_ratio:
            self._split_validation(validation_ratio)

        return None

    def augment(self, config) -> tuple:
        """Applies data augmentation transformations to the dataset based on the provided configuration.

        The `config` dictionary specifies the augmentation parameters for various transformations.
        Any key not present in the dictionary or with a value of `None` will be skipped.

        Args:
            config (dict | CfgNode): A dictionary containing the augmentation parameters with the following keys:
                - 'rotation' (list[float, float]): Range of rotation angles in degrees (e.g., [-30, 30]).
                - 'translation' (list[float, float]): Maximum translation offsets for x and y axes (e.g., [4, 4]).
                - 'scale' (list[float, float]): Range for scaling factors (e.g., [0.8, 1.2]).
                - 'shear' (list[float, float, float, float]): Shearing factors as [min_x, max_x, min_y, max_y].
                - 'noise' (float): Standard deviation of Gaussian noise to be added to the data.

        Returns:
            tuple (np.ndarray, ...): A tuple containing augmented data subsets. The number of elements in the tuple 
                depends on which augmentations are applied.

        Notes: 
            Excessive or inappropriate augmentation can lead to unrealistic samples that confuse the model. 
            For MNIST:
                - Rotation: ±15° to ±30°.
                - Translation: ≤10% of the image dimensions.
                - Scaling: 0.9x to 1.1x.
            Affine transformation: pixel(x,y) -> pixel(a x + b y + c, d x + e y + f)
        """
        print(f"Data Augmentation Started...")
        print(f"Dataset size: {len(self.train_data[0])} samples")
        start_time = time.time()

        self.config = config
        images, labels = self.train_data
        num_sections = 1 # the non-augmented training data
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
                char_index = ceil(pixel * (len(block_chars) - 1))
                mapped_char = block_chars[char_index]
                #pixel = ' ' if pixel == 0 else f'{pixel:.1f}'
                print(mapped_char, end=' ')
            print()

if __name__ == "__main__":

    import os
    import random
    from yacs.config import CfgNode
    from experiments.config import get_cfg_defaults

    np.random.seed(42) # For reproducibility

    # Load default configuration
    config: CfgNode = get_cfg_defaults()['dataset']
    config.merge_from_file("./src/data/test_case.yaml")

    # Load MINST dataset
    mnist = MNISTDatasetManager(
        batch_size = config['batch_size'],
        encoder = config['encoder'],
        nlabels = config['nlabels'],
        label_offset = config['label_offset'],
    )
    mnist.load_data(
        images_filepath = config['train_images_filepath'],
        labels_filepath = config['train_labels_filepath'],
        channels = config['channels'],
        type = 'train',
    )
    mnist.load_data(
        images_filepath = config['test_images_filepath'],
        labels_filepath = config['test_labels_filepath'],
        channels = config['channels'],
        type = 'test',
    )

    # Data Augmentation
    #x_train_augmented, y_train_augmented = mnist.augment(config['augmentation'])

    # Data Prep
    mnist.prepdata(
        validation_ratio = config['validation_ratio'],
        shuffle = config['shuffle'],
        flatten = config['flatten'],
        transpose = config['transpose'],
        debug_ratio = config['debug_ratio']
    )
    """ train_data = mnist.train_data
    images = train_data[0]
    print_images(train_data[0], train_data[1], reshape=(28,28), whitebg=False) """

    # Show some random training and test images 
    images, titles = [], []

    NUM_IMAGES = 20
    NUM_COLS = 5
    rows = int(NUM_IMAGES/NUM_COLS) + 1

    augmentations = [cfg for cfg in config['augmentation'] if config['augmentation'][cfg]]
    augmentations.insert(0, 'normal')

    indexes = list(random.randint(1, mnist.train_data[0].shape[0]//len(augmentations)) for _ in range(0, NUM_IMAGES))

    for aug_name in augmentations:
        for i in indexes:
            images.append(mnist.train_data[0][i])
            titles.append(f'Train img [{i}] = {np.argmax(mnist.train_data[1][i])}')

        plot_path = os.path.join(config['plot_filepath'], f'{config['name']}_train_{aug_name}.png')
        plot_images(plot_path, images, titles, rows, NUM_COLS, reshape=(28,28), cmap=plt.cm.spring)
        print_images(images, titles, reshape=(28,28), whitebg=False)

    indexes = list(random.randint(1, mnist.test_data[0].shape[0]) for _ in range(0, NUM_IMAGES))
    x_section, y_section = mnist.test_data
    images, titles = [], []
    for i in indexes:
        images.append(x_section[i])
        titles.append(f'Test image [{i}] = {y_section[i]}')
    plot_path = os.path.join(config['plot_filepath'], f'{config['name']}_test.png')
    plot_images(plot_path, images, titles, rows, NUM_COLS, reshape=(28,28), cmap=plt.cm.spring)