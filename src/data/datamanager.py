import numpy as np
import pandas as pd
import math

class DatasetManager:
    def __init__(
            self,
            batch_size: int,
            sequence_length: int,
            sliding_window: int,
            is_full_sequence = False,
        ):
        self.batch_size = batch_size
        self.sliding_window = sliding_window
        self.sequence_length = sequence_length

        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.mode = 'training'
        self.is_full_sequence = is_full_sequence

    def __iter__(self):
        """Generates iterable mini-batches of training data."""
        match self.mode:
            case 'training':
                x_data, y_data = self.train_data
            case 'validation':
                x_data, y_data = self.validation_data
            case 'test':
                x_data, y_data = self.test_data

        # Iterate over a divisible (mod 0) dataset length; the last idx should `batch_size`` indices less minus remainder
        num_sequences = len(x_data) - self.sequence_length
        if num_sequences < 0:
            raise RuntimeError(f"Cannot create a sequence of length {self.sequence_length} out of data with {len(x_data)} samples.")

        if self.sliding_window:
            for batch_idx in range(0, num_sequences, self.batch_size):
                x_batch = np.empty((0, self.sequence_length, x_data.shape[1]))
                y_batch = np.empty((0, self.sequence_length, y_data.shape[1]))
                for seq_idx in range(self.batch_size):
                    idx = batch_idx + seq_idx
                    if idx >= num_sequences:
                        break
                    x_seq_idx = np.arange(idx, idx + self.sequence_length)
                    y_seq_idx = np.arange(idx, idx + self.sequence_length) + 1 # shift to predict next step
                    x_seq = x_data[np.newaxis, x_seq_idx, :]
                    y_seq = y_data[np.newaxis, y_seq_idx, :]
                    x_batch = np.concatenate([x_batch, x_seq], axis=0)
                    y_batch = np.concatenate([y_batch, y_seq], axis=0)
                y_batch = y_batch if self.is_full_sequence else y_batch[:, -1, np.newaxis]
                yield x_batch, y_batch

    def load_data(
            self,
            filepath = None,
            features = None,
            target = None,
            fillna = False,
            train_ratio = 0.8,
            val_ratio = 0.1,
        ) -> None:
        """Loads datasets from filepath."""
        df = pd.read_csv(filepath)
        # Manage NAs
        df = df.ffill() if fillna else df.dropna()
        features, target = df.iloc[:, features].to_numpy(), df.iloc[:, target].to_numpy()

        # Split datasets according to train:validation:test percents
        train_end = int(len(df) * train_ratio)
        val_end = train_end + int(train_end * val_ratio)

        self.train_data = features[:train_end], target[:train_end]
        self.validation_data = features[train_end:val_end], target[train_end:val_end]
        self.test_data = features[val_end:], target[val_end:]

        return None

    def prepdata(self) -> None:
        """Prepares and preprocesses the training, validation, and test datasets.

        Normalizes values.
        """
        for dataset_name in ["train_data", "validation_data", "test_data"]:
            # Access dataset dynamically
            x_data, y_data = getattr(self, dataset_name)

            # Normalize
            x_data = (x_data - np.mean(x_data, axis=0)) / np.std(x_data, axis=0)
            y_data = (y_data - np.mean(y_data, axis=0)) / np.std(y_data, axis=0)

            # Reassign dynamically back to the original dataset
            setattr(self, dataset_name, (x_data, y_data))
        return None

if __name__ == "__main__":

    dm = DatasetManager(
        batch_size = 9,
        sequence_length = 2,
        sliding_window = True,
    )
    dm.load_data(
        filepath = './data/time_series/weather/clean_weather_small.csv',
        features = [1, 2, 3],
        target = [4],
        fillna = True,
    )
    print(dm.train_data[0].shape, dm.validation_data[0].shape, dm.test_data[0].shape, sep=', ')
    #dm.prepdata()

    print("Training")
    dm.mode = 'training'
    for i, (x_train_batch, y_train_batch) in enumerate(dm):
        print(i, x_train_batch.shape, sep=':', end=', ')

    print(f"\nValidation")
    dm.mode = 'validation'
    for i, (x_val_batch, y_val_batch) in enumerate(dm):
        print(i, x_val_batch.shape, sep=':', end=', ')
