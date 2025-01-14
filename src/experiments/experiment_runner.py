import numpy as np
import json
import os
from math import ceil

from model.basemodel import BaseModel
from data.mnist_data import MNISTDatasetManager
from data.encoders import OneHotEncoder, SmoothLabelEncoder
from optimizers.schedulers import WarmUpScheduler, StepDecayScheduler, ExponentialDecayScheduler, CosineAnnealingScheduler

from layers.denselayer import DenseLayer
from layers.batchnorm import BatchNorm
from layers.activations import Sigmoid, Tanh, ReLU, SoftMax
from layers.regularizations import Dropout

LAYERS = {
    'dense': DenseLayer,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU,
    'softmax': SoftMax,
    'batchnorm': BatchNorm,
    'dropout': Dropout
}

SCHEDULERS = {
    'warmup': WarmUpScheduler,
    'step': StepDecayScheduler,
    'exponential': ExponentialDecayScheduler,
    'cosine': CosineAnnealingScheduler
}

ENCODERS = {
    'onehot': OneHotEncoder,
    'smoothlabel': SmoothLabelEncoder
}

class ExperimentRunner:
    def __init__(self, model: BaseModel, datamanager: MNISTDatasetManager, config: dict):
        self.config = config
        # Init Data Manager
        self.datamanager: MNISTDatasetManager = datamanager(
            self.config['dataset']['batch_size'],
            ENCODERS[self.config['dataset']['encoder']]()
        )
        # Load Datasets
        self.datamanager.load_data(
            self.config['dataset']['train_images_filepath'],
            self.config['dataset']['train_labels_filepath'],
            type = 'train'
        )
        self.datamanager.load_data(
            self.config['dataset']['test_images_filepath'],
            self.config['dataset']['test_labels_filepath'],
            type = 'test'
        )
        # Prep Data
        self.datamanager.prepdata(
            type = 'train',
            shuffle = self.config['dataset']['shuffle_train_set'],
            validation_len = self.config['dataset']['validation_set_length']
        )
        self.test_data = self.datamanager.prepdata(
            type = 'test',
            shuffle = self.config['dataset']['shuffle_test_set']
        )
        # Scheduler
        steps_per_epoch = ceil(self.datamanager.train_data[0].shape[0] / self.datamanager.batch_size)
        step_size = ceil(steps_per_epoch*config['scheduler']['base']['step_ratio'])
        basemodel = SCHEDULERS[config['scheduler']['base']['name']](
            config['scheduler']['base']['learning_rate'],
            step_size,
            config['scheduler']['base']['decay_factor']
        )
        if config['scheduler']['main']['name'] == 'warmup':
            steps_total = steps_per_epoch * config['epochs']
            self.scheduler = SCHEDULERS['warmup'](
                basemodel,
                config['scheduler']['main']['learning_rate_start'],
                config['scheduler']['main']['learning_rate'],
                config['scheduler']['main']['warmup_ratio'] * steps_total
            )
        else:
            self.scheduler = basemodel

        # Init Model
        self.model: BaseModel = model()
        for layer in self.config['layers']:
            match layer['name']:
                case 'dense':
                    self.model.add(
                        LAYERS[layer['name']](
                            layer['input'],
                            layer['output'],
                            layer['weight_init']
                        )
                    )
                case 'batchnorm':
                    self.model.add(
                        LAYERS[layer['name']](
                            layer['dim']
                        )
                    )
                case 'dropout':
                    self.model.add(
                        LAYERS[layer['name']](
                            layer['rate']
                        )
                    )
                case _:
                    self.model.add(LAYERS[layer['name']]())

    def run(self) -> None:
        """Runs an experiment for a given configuration."""
        # Train Model
        results = self.model.train(
            self.datamanager,
            self.scheduler,
            self.config['epochs']
        )
        # Evaluate
        test_accuracy = self.evaluate(self.model, self.test_data)

        # Log Results
        self.log_results(results, test_accuracy)

    def evaluate(self, model: BaseModel, test_data: np.ndarray):
        """Evaluates the model on the test dataset."""
        # Inference
        test_probabilities = model.forward(test_data[0], is_training=False)
        test_predictions = np.argmax(test_probabilities, axis=1)
        # Calculate Accuracy
        test_accuracy = np.mean(test_predictions == test_data[1])
        return test_accuracy

    def _create_model_name(self):
        """Creates model name based on architecture
        
        Example: mlp_model[784-256-256-10]
        """
        model_name = f'mlp_model[{self.config['input_layer']}'
        model_name += ''.join(f'-{hl}' for hl in self.config['hidden_layers'])
        model_name += f'-{self.config['output_layer']}]'
        return model_name

    def log_results(self, train_results, test_accuracy):
        """Logs the results of the experiment."""
        #model_name = self._create_model_name()
        model_name = self.model.__str__()
        print(f"\n{model_name}, Epochs: {self.config['epochs']}, Batch size: {self.config['dataset']['batch_size']}, " +
                f"Learning rate: {self.config['scheduler']['base']['learning_rate']} \
                \n{"─" * 15} Loss {"─" * 20} \
                \nTraining Loss:\t{train_results['training_losses'][-1]:.3} \
                \nValid Loss:\t{train_results['validation_losses'][-1]:.3} \
                \n{"─" * 15} Accuracies {"─" * 15} \
                \nTraining Acc.:\t{train_results['training_accuracies'][-1]:.3%} \
                \nValid Acc.:\t{train_results['validation_accuracies'][-1]:.3%} \
                \nTest Acc.:\t{test_accuracy:.3%}\n")        
        
        experiment_filepath = os.path.join(
            self.config['log_filepath'], 
            model_name
        )
        if not os.path.exists(experiment_filepath):
            os.makedirs(experiment_filepath)

        experiment_params = f"_e{self.config['epochs']}_b{self.config['dataset']['batch_size']}_lr{self.config['scheduler']['base']['learning_rate']:.2}"
        experiment_name = 'experiment_' + experiment_params + '.json'
        experiment_filepath = os.path.join(
            experiment_filepath,
            experiment_name
        )
        with open(experiment_filepath, 'w') as f:
            json.dump({
                'training_losses': train_results['training_losses'],
                'training_accuracies': train_results['training_accuracies'],
                'validation_losses': train_results['validation_losses'],
                'validation_accuracies': train_results['validation_accuracies']
                }, f)
