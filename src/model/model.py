import numpy as np
from math import ceil
from tqdm import tqdm, trange
import os
import pickle

from data.mnist_data import MNISTDatasetManager
from optimizers.schedulers import Scheduler
from optimizers.optimizer import Optimizer
from optimizers.optimizer_factory import OptimizerFactory
from layers.layer import Layer
from layers.dense import Dense
from losses.losses import Loss, LOSS_FN
from metrics.metrics import Metric, MetricManager, METRICS

class Model:
    """Model base class.
    """
    def __init__(self, name: str = None):
        self.layers: list[Layer] = []
        self.training_metrics = []
        self.training_losses = []
        self.validation_metrics = []
        self.validation_losses = []
        self.name = name if name is not None else self.__class__.__name__

        # Compile attributes
        self.is_compiled = False
        self.loss_fn = None
        self.optimizer = None
        self.metrics: MetricManager = None

    def add(self, layer: Layer):
        """Adds a layer to the model's architecture

        This method appends a new layer to the model's list of layers, allowing for the 
        sequential construction of the model's architecture. Layers are processed in the 
        order they are added.

        #### Args
        - `layer` (`Layer`): The layer to be added to the model.
        """
        self.layers.append(layer)

    def compile(
        self,
        optimizer: str | Optimizer = 'sgd',
        loss: str | Loss = 'cross-entropy-loss',
        metrics = None,
    ) -> None:
        """Configures the model for training.

        This method assigns the specified optimizer to the model and initializes 
        any necessary optimization-related parameters for trainable layers.

        #### Args:
            - optimizer (Optimizer): The optimization algorithm to be used to update the model's parameters (default = 'sgd').
            - `loss_fn` (`str` | `Loss`): The loss function to be used to calculate the predictive error of the model (default = 'cross-entropy-loss').
        """
        # Optimizer
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            optimizer_factory = OptimizerFactory()
            self.optimizer = optimizer_factory.create(optimizer)
        for layer in self.layers:
            if layer.trainable_params is not None:
                self.optimizer.init_params(layer.trainable_params)
        # Loss
        if isinstance(loss, Loss):
            self.loss_fn = loss
        else:
            self.loss_fn: Loss = LOSS_FN[loss]()

        # Metrics
        if metrics is not None and isinstance(metrics, (str, list, tuple)):
            self.metrics = MetricManager(metrics)

        self.is_compiled = True
        return None

    def compute_metrics(self, x, y, y_pred):
        pass

    def _add_padding(self, info: str, column_span: int = 20):
        return " " * (column_span - len(str(info)))

    def summary(self):
        total_model_params = 0
        print(f"Model: {self.name}")
        print(f"{"─" * 80}")
        print("Layer (type)" + self._add_padding("Layer (type)") +
              "Input Shape" + self._add_padding("Input Shape") +
              "# Parameters" + self._add_padding("# Parameters"))
        print(f"{"=" * 80}")
        for layer in self.layers:
            shape = layer.shape if layer.shape is not None else "-"
            print(layer.name + self._add_padding(layer.name) +
                  f"{shape:}" + self._add_padding(shape) + 
                  f"{layer.total_params:,}" + self._add_padding(layer.total_params))
            print(f"{" " * 80}")
            total_model_params += layer.total_params
        print(f"{"=" * 80}")
        print(f"Total Trainable Params: {total_model_params:,}")
        print(f"{"─" * 80}")

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
            - `grad` (`np.ndarray`): The gradient w.r.t. to the loss
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self) -> None:
        """Performs the update pass through the network.

        This method iterates through the layers with trainable parameters and updates 
        their parameters using the computed gradients. It applies any available optimization 
        strategy to refine the gradients before updating the layer parameters.
        """
        for idx, layer in enumerate(self.layers):
            if layer.trainable_params is not None:
                if self.optimizer is not None:
                    computed_gradients = self.optimizer.update(idx, layer.gradients)
                else:
                    computed_gradients = layer.gradients
                layer.update(self.learning_rate, computed_gradients)
        return None

    def train(self, datamanager: MNISTDatasetManager, scheduler: Scheduler, epochs: int, start_epoch: int = 0, checkpoint: list = None) -> dict:
        """Trains the MLP on the training data.

        Performs forward and backward passes at a given learning rate, and over a number of epochs.

        #### Args
            - `datamanager` (`MNISTDatasetManager`): A DataManager class containing the training and validation data, as well as an iterator for mini-batch.
            - `scheduler` (`Scheduler`): The scheduler that will implement the learning rate update strategy during training.
            - `epochs` (int): The number of times the model will iterate over the entire training dataset.
            - `start_epoch` (`int`): 
            - `checkpoint` (`list`): Checkpoint settings (default = `None`).

        #### Returns
            - dict: Dictionary containing the following:
                - `'weights'` (`list[np.ndarray]`): Final weights of the model.
                - `'bias'` (`list[np.ndarray]`): Final biases of the model.
                - `'training_metrics'` (`list[float]`): Training metric values recorded at each epoch.
                - `'training_losses'` (`list[float]`): Training loss values recorded at each epoch.
        """
        self.epochs = epochs - start_epoch
        self.datamanager = datamanager
        self.scheduler = scheduler
        val_loss = [0.0]

        if not self.is_compiled:
            raise RuntimeError("Can't start training on an uncompiled model.")

        with trange(self.epochs) as t:
            for epoch in t:
                batch_losses = []
                self.metrics.init_results('batch')

                self.current_epoch = start_epoch + epoch + 1 # used to track checkpoint's epoch. Offset required to skip 0 index.

                total_batches = ceil(datamanager.train_data[0].shape[0] / datamanager.batch_size)
                datamanager.mode = 'training'
                for batch_idx, (X_batch, y_batch) in enumerate(datamanager):
                    t.set_description(f"Epoch {start_epoch + epoch+1} ({batch_idx+1}/{total_batches})") # Monitor epoch and batch progress in terminal

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
                    self.update()

                    # Monitor batch metrics
                    if batch_idx % datamanager.batch_size == 0:
                        self.metrics.compute(y_hat, y_batch, 'batch')
                        batch_losses.append(loss)
                        monitor_dict = {'tLoss': f"{loss:.2f}", 'vLoss': f"{val_loss[-1]:.2f}"}
                        for metric in self.metrics.metrics:
                            t_key_name = 't'+metric.name.capitalize()
                            v_key_name = 'v'+metric.name.capitalize()
                            monitor_dict[t_key_name] = f"{self.metrics.get_latest_result('batch', metric.name):.2f}"
                            monitor_dict[v_key_name] = f"{self.metrics.get_latest_result('validation', metric.name):.2f}"
                        t.set_postfix(monitor_dict)

                    scheduler.step()

                # Monitor epoch metrics
                self.metrics.compute(y_hat, y_batch, 'training')
                self.training_losses.append(loss)

                # Monitor Validation Metrics & Loss
                if datamanager.validation_data:
                    datamanager.mode = 'validation'
                    _, val_loss = self.evaluate(datamanager)
                    self.validation_losses = val_loss

                # Checkpoint
                if checkpoint and (epoch+1) % checkpoint[1] == 0 and epoch > 0:
                    self.save_checkpoint(checkpoint[0])

                # Monitoring Metrics
                t.refresh()

        return {
            'training_metrics': self.metrics.results['training'],
            'training_losses': self.training_losses,
            'validation_metrics': self.metrics.results['validation'],
            'validation_losses': self.validation_losses,
        }

    def evaluate(self, data: tuple[np.ndarray, np.ndarray], batch_size: int = None):
        losses = []

        if batch_size:
            data.batch_size = batch_size

        for X_batch, y_batch in data:
            y_hat_batch = self.forward(X_batch, is_training=False)
            self.metrics.compute(y_hat_batch, y_batch, data.mode)
            loss = self.loss_fn.forward(y_hat_batch, y_batch)
            losses.append(loss)

        return self.metrics.results[data.mode], losses

    def predict(self, data):
        y_pred = []
        for X_batch, _ in data:
            y_hat_batch = self.forward(X_batch, is_training=False)
            y_pred.append(y_hat_batch)
        return np.vstack(y_pred)

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

        self.training_metrics = nn_model.training_metrics
        self.training_losses = nn_model.training_losses
        self.validation_metrics = nn_model.validation_metrics
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
        'nlabels': 10,
        'batch_size': 64,
        'metrics_filepath': './plots/metrics/',
        'checkpoint_filepath': './results/checkpoints/',
        'checkpoint_epoch_freq': 2,
        'load_checkpoint': 'model[784-800-10]/model[784-800-10]_e2of4_b64.pkl'
    }

    # Load MINST dataset
    batch_size = config['batch_size']
    mnist = MNISTDatasetManager(
        batch_size = batch_size,
        nlabels = config['nlabels'],
    )

    mnist.load_data(
        config['train_images_filepath'],
        config['train_labels_filepath'],
        type = 'train',
    )
    mnist.load_data(
        config['test_images_filepath'],
        config['test_labels_filepath'],
        type = 'test'
    )

    train_data = mnist.prepdata()

    epochs = 4
    learning_rate = 9e-2
    learning_rate_start = 1e-3

    # Scheduler
    steps_per_epoch = ceil(mnist.train_data[0].shape[0] / batch_size)
    steps_total = steps_per_epoch * epochs
    basemodel = StepDecayScheduler(learning_rate, step_size=ceil(steps_per_epoch*.15), decay_factor=0.90)
    scheduler = WarmUpScheduler(basemodel, learning_rate_start, learning_rate, steps_total*0.1)
    #plot_schedule(scheduler, epochs, steps_per_epoch) # Debug

    # Setup NN
    mlp = Model(name="my_mlp")

    # Build model by adding lñayers sequentially
    mlp.add(Dense((784, 800), weight_init='he'))
    mlp.add(ReLU())
    mlp.add(Dropout(0.3))
    mlp.add(Dense((800, 10), weight_init='xavier'))
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
    test_probabilities = mlp.forward(mnist.test_data[0], is_training=False)
    test_predictions = np.argmax(test_probabilities, axis=1)

    # Accuracy
    test_accuracy = np.mean(test_predictions == mnist.test_data[1])

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