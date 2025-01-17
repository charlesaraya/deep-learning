import numpy as np
from abc import ABC, abstractmethod

class Encoder:
    """Interface for encoding functions."""
    def __init__(self, nlabels: int, label_offset: int = 0):
        self.nlabels = nlabels
        self.label_offset = label_offset

    @abstractmethod
    def encode(self, labels: np.ndarray):
        """Encode labels"""
        pass

class OneHotEncoder(Encoder):
    """One-hot encoder class."""
    def __init__(self, nlabels: int, label_offset: int = 0):
        super(OneHotEncoder, self).__init__(nlabels, label_offset)

    def encode(self, labels: np.ndarray):
        """Encodes each target label class into its one-hot format.

        Args:
            labels (ndarray): Array of integers representing the target labels.
        """
        labels_encoded  = []
        labels = labels - self.label_offset # ensures labels are indexed at 0 (i.e EMNIST 1:26)
        for label in labels:
            labels_encoded.append(np.zeros(self.nlabels))
            labels_encoded[-1][label] = 1
        return np.asarray(labels_encoded)

class SmoothLabelEncoder(Encoder):
    """Smooth Label Encoder class"""
    def __init__(self, nlabels: int, label_offset: int = 0, smoothing_factor: float = 0.1):
        super(SmoothLabelEncoder, self).__init__(nlabels)
        self.smoothing_factor = smoothing_factor

    def encode(self, labels: np.ndarray):
        """Returns the softened one-hot encoding of each label class.

        Args:
            labels (ndarray): Array of integers representing the target labels.
        """
        labels_encoded  = []
        labels = labels - self.label_offset
        for label in labels:
            encoding = self.smoothing_factor / self.nlabels * np.ones(self.nlabels)
            encoding[label] = 1 - self.smoothing_factor
            labels_encoded.append(encoding)
        return np.asarray(labels_encoded)