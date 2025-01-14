import numpy as np
from math import ceil
from tqdm import tqdm, trange
import os
import pickle

from data.mnist_data import MNISTDatasetManager
from optimizers.schedulers import Scheduler, WarmUpScheduler, StepDecayScheduler, plot_schedule
from layers.layer import Layer
from layers.denselayer import DenseLayer
from layers.regularizations import Dropout
from layers.batchnorm import BatchNorm
import layers.losses as loss_fn
from layers.activations import ACTIVATION_FN, Sigmoid, Tanh, SoftMax, ReLU

class BaseModel:

    def __init__(self):
        self.layers: list[Layer] = []
        self.training_accuracies = []
        self.training_losses = []
        self.validation_accuracies = []
        self.validation_losses = []

    def add(self, layer: Layer):
        """Adds a layer to the model"""
        self.layers.append(layer)

    def __str__(self):
        self.name = f'model[{self.layers[0].shape[0]}'
        self.name += ''.join(f'-{layer.shape[1]}' for layer in self.layers if layer.shape[0] != 0)
        self.name += ']'
        return self.name
    
    def forward(self, X: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Performs the forward pass through the network.

        Args:
            X (ndarray): Input data. Each row corresponds to a sample and each column corresponds to a feature.

        Returns:
            ndarray: Output of the network. Each row corresponds to the predicted values for each sample, and each column corresponds to a target label.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output, is_training)
        return output

    def backward(self, grad: np.ndarray) -> None:
        """Performs the backward pass through the network.

        Computing the gradients of the loss w.r.t. the model parameters, and updates the weights and biases.

        Args:
            y_hat (ndarray): Predicted probabilities from the forward pass.
            y (ndarray): The true target labels for each training sample.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, datamanager: MNISTDatasetManager, scheduler: Scheduler, epochs: int, start_epoch: int = 0, checkpoint: list = None) -> dict:
        """Trains the MLP on the training data.
        
        Performs forward and backward passes at a given learning rate, and over a number of epochs.

        Args:
            datamanager (MNISTDatasetManager): A DataManager class containing the training and validation data, as well as an iterator for mini-batch.
            epochs (int): The number of times the model will iterate over the entire training dataset.
            learning_rate (float): The learning rate used for gradient descent to update the model parameters.

        Returns:
            dict: Dictionary containing the following:
            - 'weights' (list[ndarray]): Final weights of the model.
            - 'bias' (list[ndarray]): Final biases of the model.
            - 'training_accuracies' (list): Training accuracy values recorded at each epoch.
            - 'training_losses' (list): Training loss values recorded at each epoch.
        """
        self.epochs = epochs - start_epoch
        self.datamanager = datamanager
        self.scheduler = scheduler

        with trange(self.epochs) as t:
            for epoch in t:
                t.set_description(f"Epoch {start_epoch + epoch+1}") # Monitor epoch progress in terminal
                batch_accuracies, batch_losses = [], []
                self.current_epoch = start_epoch + epoch + 1 # used to track checkpoint's epoch. Offset required to skip 0 index.
                for X_batch, y_batch in datamanager:

                    self.learning_rate = scheduler.get_lr()

                    # Forward Pass
                    y_hat = self.forward(X_batch)

                    # Calculate error in prediction using Cross-Entropy Loss function
                    loss = loss_fn.cross_entropy_loss(y_hat, y_batch)
                    # Calculate gradient w.r.t loss
                    grad = (y_hat  - y_batch) / y_batch.shape[0]
                    # Backpropagation Pass: Calculate Gradients, Weights & Bias
                    self.backward(grad)

                    # Gradient Descent: Update Weights and Biases
                    for layer in self.layers:
                        if isinstance(layer, DenseLayer):
                            layer.weights -= self.learning_rate * layer.dweights
                            layer.bias -= self.learning_rate * layer.dbias

                    # Monitor batch metrics
                    predictions = np.argmax(y_hat, axis=1)
                    accuracy = np.mean(predictions == np.argmax(y_batch, axis=1))
                    batch_accuracies.append(accuracy)
                    batch_losses.append(loss)

                    scheduler.step()

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
                    val_loss = loss_fn.cross_entropy_loss(val_probabilities, datamanager.validation_data[1])
                    self.validation_losses.append(val_loss)

                # Checkpoint
                if checkpoint and (epoch+1) % checkpoint[1] == 0 and epoch > 0:
                    self.save_checkpoint(checkpoint[0], self.current_epoch)

                # Monitoring Metrics
                t.set_postfix(
                    tLoss = epoch_loss,
                    tAcc = epoch_accuracy*100,
                    vLoss = val_loss,
                    vAcc = val_accuracy*100
                )

        self.weights, self.biases = [], []
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                self.weights.append(layer.weights)
                self.biases.append(layer.bias)

        return {
            'weights': self.weights,
            'bias': self.biases,
            'training_accuracies': self.training_accuracies,
            'training_losses': self.training_losses,
            'validation_accuracies': self.validation_accuracies,
            'validation_losses': self.validation_losses
        }

    def load_checkpoint(self, filepath: str):
        """
        Load serialized model with weights and biases.

        Args:
            filepath (str): Filepath to the model checkpoint.
        """
        with open(filepath,'rb') as f:
            nn_model: BaseModel = pickle.load(f, encoding='bytes')
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

    def save_checkpoint(self, directory: str, current_epoch):
        """
        Save serialized model of neural network.

        Args:
            directory (str): Directory name for the model checkpoint.
        """
        self.random_state = np.random.get_state()

        model_name = self.__str__()
        model_details = f"{model_name}_e{current_epoch}of{self.epochs}_b{self.datamanager.batch_size}.pkl"

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
    mnist = MNISTDatasetManager(batch_size)

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
    mlp = BaseModel()
    
    """ # Option 1
    mlp.add(DenseLayer(input_layer, 64, activation='tanh'))
    mlp.add(DenseLayer(64, output_layer, activation='softmax')) """

    # Option 2
    mlp.add(DenseLayer(input_layer, 800, weight_init='he'))
    mlp.add(ReLU())
    mlp.add(Dropout(0.3))
    mlp.add(DenseLayer(800, output_layer, weight_init='xavier'))
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

    mlp2 = BaseModel()
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