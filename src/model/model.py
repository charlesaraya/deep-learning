import numpy as np
from math import ceil
from tqdm import tqdm, trange
import os
import pickle

from data.mnist_data import MNISTDatasetManager
from optimizers.schedulers import Scheduler
from layers.layer import Layer
from layers.dense import Dense
from losses.losses import Loss, LOSS_FN

class Model:
    """Model base class.
    """
    def __init__(self, name: str = None):
        self.layers: list[Layer] = []
        self.training_accuracies = []
        self.training_losses = []
        self.validation_accuracies = []
        self.validation_losses = []
        self.name = name if name is not None else self.__class__.__name__

    def add(self, layer: Layer):
        """Adds a layer to the model's architecture

        This method appends a new layer to the model's list of layers, allowing for the 
        sequential construction of the model's architecture. Layers are processed in the 
        order they are added.

        #### Args
        - `layer` (`Layer`): The layer to be added to the model.
        """
        self.layers.append(layer)

    def summary(self):
        print(f"Model: {self.name}")
        print(f"{"─" * 50}")
        print(f"Layer (type)")
        print(f"{"=" * 50}")
        for layer in self.layers:
            print(f"{layer.name} ({layer.__class__.__name__})")
            print(f"{"─" * 50}")

    def __str__(self):
        self.name = f'model[{self.layers[0].shape[0]}'
        self.name += ''.join(f'-{layer.shape[1]}' for layer in self.layers if isinstance(layer, Dense))
        self.name += ']'
        return self.name
    
    def forward(self, X: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Performs the forward pass through the network.

        #### Args
            - `X` (`np.ndarray`): The input data. Each row corresponds to a sample and each column corresponds to a feature.

        #### Returns
            - `np.ndarray`: The output of the network. Each row corresponds to the predicted values for each sample, and each column corresponds to a target label.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output, is_training)
        return output

    def backward(self, grad: np.ndarray) -> None:
        """Performs the backward pass through the network.

        Computing the gradients of the loss w.r.t. the model parameters, and updates the weights and biases.

        #### Args
            - `y_hat` (`np.ndarray`): Predicted probabilities from the forward pass.
            - `y` (`np.ndarray`): The true target labels for each training sample.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, datamanager: MNISTDatasetManager, scheduler: Scheduler, epochs: int, loss_fn: str, start_epoch: int = 0, checkpoint: list = None) -> dict:
        """Trains the MLP on the training data.

        Performs forward and backward passes at a given learning rate, and over a number of epochs.

        #### Args
            - `datamanager` (`MNISTDatasetManager`): A DataManager class containing the training and validation data, as well as an iterator for mini-batch.
            - `epochs` (int): The number of times the model will iterate over the entire training dataset.
            - `loss_fn` (`str`): The loss function to be used to calculate the predictive error of the model.
            - `start_epoch` (`int`): 
            - `checkpoint` (`list`): Checkpoint settings (default = `None`).

        #### Returns
            - dict: Dictionary containing the following:
                - `'weights'` (`list[np.ndarray]`): Final weights of the model.
                - `'bias'` (`list[np.ndarray]`): Final biases of the model.
                - `'training_accuracies'` (`list[float]`): Training accuracy values recorded at each epoch.
                - `'training_losses'` (`list[float]`): Training loss values recorded at each epoch.
        """
        self.epochs = epochs - start_epoch
        self.datamanager = datamanager
        self.scheduler = scheduler
        self.loss_fn: Loss = LOSS_FN[loss_fn]()

        with trange(self.epochs) as t:
            for epoch in t:
                t.set_description(f"Epoch {start_epoch + epoch+1}") # Monitor epoch progress in terminal
                batch_accuracies, batch_losses = [], []
                self.current_epoch = start_epoch + epoch + 1 # used to track checkpoint's epoch. Offset required to skip 0 index.

                i =   0
                total_batches = ceil(datamanager.train_data[0].shape[0] / datamanager.batch_size)
                with tqdm(total=total_batches) as b:
                    for X_batch, y_batch in tqdm(datamanager):
                        b.set_description(f"Batch {i+1}") # Monitor batch progress in terminal
                        self.learning_rate = scheduler.get_lr()

                        # Forward Pass
                        y_hat = self.forward(X_batch)

                        # Calculate error in prediction
                        loss = self.loss_fn.forward(y_hat, y_batch)
                        # Calculate gradient w.r.t loss
                        grad = self.loss_fn.backward(y_hat, y_batch)
                        # Backpropagation Pass: Calculate Gradients, Weights & Bias
                        self.backward(grad)

                        # Gradient Descent: Update Learning Parameters
                        for layer in self.layers:
                            layer.update(self.learning_rate)

                        # Monitor batch metrics
                        predictions = np.argmax(y_hat, axis=1)
                        accuracy = np.mean(predictions == np.argmax(y_batch, axis=1))
                        batch_accuracies.append(accuracy)
                        batch_losses.append(loss)

                        scheduler.step()
                        # Monitoring Metrics
                        b.set_postfix(
                            BAcc = accuracy*100,
                            BLoss = loss,
                        )
                        b.update(1)
                        i += 1

                # Monitor epoch metrics
                epoch_loss = batch_losses[-1]
                epoch_accuracy = batch_accuracies[-1]
                self.training_accuracies.append(epoch_accuracy)
                self.training_losses.append(epoch_loss)

                # Validation
                if datamanager.validation_data:
                    # Accuracy
                    val_probabilities = self.forward(datamanager.validation_data[0], is_training=False)
                    val_predictions = np.argmax(val_probabilities, axis=1)
                    val_labels = np.argmax(datamanager.validation_data[1], axis=1)
                    val_accuracy = np.mean(val_predictions == val_labels)
                    self.validation_accuracies.append(val_accuracy)
                    # Loss
                    val_loss = self.loss_fn.forward(val_probabilities, datamanager.validation_data[1])
                    self.validation_losses.append(val_loss)

                # Checkpoint
                if checkpoint and (epoch+1) % checkpoint[1] == 0 and epoch > 0:
                    self.save_checkpoint(checkpoint[0])

                # Monitoring Metrics
                t.set_postfix(
                    tLoss = epoch_loss,
                    tAcc = epoch_accuracy*100,
                    vLoss = val_loss,
                    vAcc = val_accuracy*100
                )

        return {
            'training_accuracies': self.training_accuracies,
            'training_losses': self.training_losses,
            'validation_accuracies': self.validation_accuracies,
            'validation_losses': self.validation_losses
        }

    def evaluate(self, data: tuple[np.ndarray, np.ndarray], batch_size: int = None):
        if batch_size is None:
            batch_size = len(data)
        data_length = len(data)
        data_indices = np.arange(data_length)

        eval_probabilities = []
        for start_idx in range(0, len(data), batch_size):
            end_idx = start_idx + batch_size
            batch_indices = data_indices[start_idx:end_idx]
            eval_probabilities.append(self.forward(data[batch_indices], is_training=False))
        return np.vstack(eval_probabilities)

    def load_checkpoint(self, filepath: str):
        """Load serialized model with weights and biases.

        #### Args
            - `filepath` (`str`): Filepath to the model checkpoint.
        """
        with open(filepath,'rb') as f:
            nn_model: Model = pickle.load(f, encoding='bytes')
        f.close()

        np.random.set_state(nn_model.random_state)
        self.layers = nn_model.layers

        self.training_accuracies = nn_model.training_accuracies
        self.training_losses = nn_model.training_losses
        self.validation_accuracies = nn_model.validation_accuracies
        self.validation_losses = nn_model.validation_losses

        self.scheduler = nn_model.scheduler
        self.datamanager = nn_model.datamanager
        self.epochs = nn_model.epochs
        self.current_epoch = nn_model.current_epoch

    def save_checkpoint(self, directory: str):
        """Save serialized model of neural network.

        #### Args
            - `directory` (`str`): Directory name for the model checkpoint.
        """
        self.random_state = np.random.get_state()

        model_name = self.__str__()
        model_details = f"{model_name}_e{self.current_epoch}of{self.epochs}_b{self.datamanager.batch_size}.pkl"

        modelpath = os.path.join(directory, model_name)
        filepath = os.path.join(modelpath, model_details)

        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(modelpath):
            os.makedirs(modelpath)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        f.close()

if __name__ == "__main__":

    from data.mnist_data import MNISTDatasetManager
    from data.encoders import OneHotEncoder, SmoothLabelEncoder
    from optimizers.schedulers import Scheduler, WarmUpScheduler, StepDecayScheduler, plot_schedule
    from layers.layer import Layer
    from layers.dense import Dense
    from layers.dropout import Dropout
    from layers.batchnorm import BatchNorm
    from losses.losses import Loss, LOSS_FN
    from layers.activations import ACTIVATION_FN, Sigmoid, Tanh, SoftMax, ReLU

    # Set file paths based on added MNIST Datasets
    config = {
        'train_images_filepath': './data/MNIST/train-images',
        'train_labels_filepath': './data/MNIST/train-labels',
        'test_images_filepath': './data/MNIST/test-images',
        'test_labels_filepath': './data/MNIST/test-labels',
        'metrics_filepath': './plots/metrics/',
        'checkpoint_filepath': './results/checkpoints/',
        'checkpoint_epoch_freq': 2,
        'load_checkpoint': 'model[784-800-10]/model[784-800-10]_e2of4_b64.pkl'
    }

    # Load MINST dataset
    batch_size = 64
    mnist = MNISTDatasetManager(batch_size, SmoothLabelEncoder())

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

    train_data = mnist.prepdata('train', shuffle=True, validation_len=10000)
    test_data = mnist.prepdata('test')

    # Architecture
    input_layer = train_data[0].shape[1]
    hidden_layer = [512]
    output_layer = train_data[1].shape[1]

    epochs = 4
    learning_rate = 9e-2
    learning_rate_start = 1e-3

    # Scheduler
    steps_per_epoch = ceil(train_data[0].shape[0] / batch_size)
    steps_total = steps_per_epoch * epochs
    basemodel = StepDecayScheduler(learning_rate, step_size=ceil(steps_per_epoch*.15), decay_factor=0.90)
    scheduler = WarmUpScheduler(basemodel, learning_rate_start, learning_rate, steps_total*0.1)
    #plot_schedule(scheduler, epochs, steps_per_epoch) # Debug

    # Setup NN
    mlp = Model()
    
    """ # Option 1
    mlp.add(DenseLayer(input_layer, 64, activation='tanh'))
    mlp.add(DenseLayer(64, output_layer, activation='softmax')) """

    # Option 2
    mlp.add(Dense(input_layer, 800, weight_init='he'))
    mlp.add(ReLU())
    mlp.add(Dropout(0.3))
    mlp.add(Dense(800, output_layer, weight_init='xavier'))
    mlp.add(SoftMax())

    # Train
    output = mlp.train(
        mnist,
        scheduler,
        epochs,
        checkpoint = [
            config['checkpoint_filepath'],
            config['checkpoint_epoch_freq']
        ]
    )

    # Inference
    test_probabilities = mlp.forward(test_data[0], is_training=False)
    test_predictions = np.argmax(test_probabilities, axis=1)

    # Accuracy
    test_accuracy = np.mean(test_predictions == test_data[1])

    # Results
    print(f"\n{mlp.__str__()}, Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate} \
            \n{"─" * 15} Loss {"─" * 20} \
            \nTraining Loss:\t{output['training_losses'][-1]:.3} \
            \nValid Loss:\t{output['validation_losses'][-1]:.3} \
            \n{"─" * 15} Accuracies {"─" * 15} \
            \nTraining Acc.:\t{output['training_accuracies'][-1]:.3%} \
            \nValid Acc.:\t{output['validation_accuracies'][-1]:.3%} \
            \nTest Acc.:\t{test_accuracy:.3%}\n")

    # Load Checkpoint
    mlp2 = Model()
    checkpoint_path = os.path.join(config['checkpoint_filepath'], config['load_checkpoint'])
    mlp2.load_checkpoint(checkpoint_path)
    output = mlp2.train(mlp2.datamanager, mlp2.scheduler, mlp2.epochs, mlp2.current_epoch)

    # Results after checkpoint
    print(f"\n{mlp.__str__()}, Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate} \
            \n{"─" * 15} Loss {"─" * 20} \
            \nTraining Loss:\t{output['training_losses'][-1]:.3} \
            \nValid Loss:\t{output['validation_losses'][-1]:.3} \
            \n{"─" * 15} Accuracies {"─" * 15} \
            \nTraining Acc.:\t{output['training_accuracies'][-1]:.3%} \
            \nValid Acc.:\t{output['validation_accuracies'][-1]:.3%} \
            \nTest Acc.:\t{test_accuracy:.3%}\n")