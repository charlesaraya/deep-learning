import numpy as np
from abc import ABC, abstractmethod

class Encoder:
    """Interface for encoding functions."""
    def __init__(self):
        self.labels = None
        self.nlabels = 0

    @abstractmethod
    def encode(self, labels: np.ndarray):
        """Encode labels"""
        pass

class OneHotEncoder(Encoder):
    """One-hot encoder class."""
    def __init__(self):
        super(OneHotEncoder, self).__init__()

    def encode(self, labels: np.ndarray):
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

class SmoothLabelEncoder(Encoder):
    """Smooth Label Encoder class"""
    def __init__(self, smoothing_factor: float = 0.1):
        super(SmoothLabelEncoder, self).__init__()
        self.smoothing_factor = smoothing_factor

    def encode(self, labels: np.ndarray):
        """Returns the softened one-hot encoding of each label class.

        Args:
            labels (ndarray): Array of integers representing the target labels.
        """
        nlabels = max(labels) + 1
        labels_encoded  = []
        for label in labels:
            encoding = self.smoothing_factor / nlabels * np.ones(nlabels)
            encoding[label] = 1 - self.smoothing_factor
            labels_encoded.append(encoding)
        return np.asarray(labels_encoded)