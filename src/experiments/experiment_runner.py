import numpy as np
import json
import os
from math import ceil

from model.model import Model
from data.mnist_data import MNISTDatasetManager
from optimizers.scheduler_factory import SchedulerFactory
from layers.layer_factory import LayerFactory

class ExperimentRunner:
    def __init__(self, model: Model, datamanager: MNISTDatasetManager, config: dict):
        self.config = config

        # Init Data Manager
        self.datamanager: MNISTDatasetManager = datamanager(
            self.config['dataset']['batch_size'],
            self.config['dataset']['encoder'],
            self.config['dataset']['nlabels'],
            self.config['dataset']['label_offset'],
        )
        # Load Datasets
        self.datamanager.load_data(
            self.config['dataset']['train_images_filepath'],
            self.config['dataset']['train_labels_filepath'],
            channels = self.config['dataset']['channels'],
            type = 'train',
        )
        self.datamanager.load_data(
            self.config['dataset']['test_images_filepath'],
            self.config['dataset']['test_labels_filepath'],
            channels = self.config['dataset']['channels'],
            type = 'test'
        )
        # Data Augmentation
        self.datamanager.augment(config['dataset']['augmentation'])

        # Prep Data
        self.datamanager.prepdata(
            validation_ratio = self.config['dataset']['validation_ratio'],
            shuffle = self.config['dataset']['shuffle'],
            flatten = self.config['dataset']['flatten'],
            transpose = self.config['dataset']['transpose'],
            debug_ratio = self.config['dataset']['debug_ratio'],
        )
        # Scheduler
        scheduler_factory = SchedulerFactory(
            dataset_len = self.datamanager.train_data[0].shape[0],
            batch_size = self.datamanager.batch_size
        )
        self.scheduler = scheduler_factory.create(self.config['scheduler'])

        # Init Model
        layer_factory = LayerFactory()
        self.model: Model = model(self.config['model']['name'])
        for layer in self.config['layers']:
            self.model.add(layer_factory.create(layer))

        if self.config['model']['name']:
            self.model.summary()

    def run(self) -> None:
        """Runs an experiment for a given configuration."""
        # Train Model
        results = self.model.train(
            self.datamanager,
            self.scheduler,
            self.config['epochs'],
            self.config['loss_fn'],
            checkpoint = [
                self.config['checkpoint']['filepath'],
                self.config['checkpoint']['epoch_freq']
            ],
        )
        # Evaluate
        test_accuracy = self.evaluate(
            batch_size = self.config['model']['batch_eval'],
            rejection_criteria = self.config['test']['evaluation']['rejection_criteria'],
        )

        # Log Results
        self.log_results(results, test_accuracy)

    def evaluate(self, batch_size = None, rejection_criteria: list[float] = None):
        """Evaluates the model on the test dataset."""
        # Inference
        test_probabilities = self.model.evaluate(self.datamanager.test_data[0], batch_size=batch_size)
        test_predictions = np.argmax(test_probabilities, axis=1) + self.config['dataset']['label_offset']

        # Calculate Rejection
        if rejection_criteria:
            self.reject(test_probabilities, test_predictions, rejection_criteria)

        # Calculate Accuracy
        test_accuracy = np.mean(test_predictions == self.datamanager.test_data[1])
        return test_accuracy

    def reject(self, test_probabilities: np.ndarray, test_predictions: np.ndarray, rejection_criteria: list[float]) -> None:
        self.rejection_metrics = True

        test_size = len(test_probabilities)
        test_mask = np.ones(test_size, dtype=bool) 

        for idx, (sample, pred) in enumerate(zip(test_probabilities, test_predictions)):
            # Criteria 1: 1st prediction is over theta1
            if sample[pred] < rejection_criteria[0]:
                test_mask[[idx]] = False

            # Criteria 2: 2nd prediction is under theta2
            mask = np.ones(sample.size, dtype=bool)
            mask[[pred]] = False
            pred_sec = np.argmax(sample[mask])
            if sample[mask][pred_sec] > rejection_criteria[1]:
                test_mask[[idx]] = False

            # Criteria 3: Diff bw 1st prediction and 2nd prediction is < theta3
            if sample[pred] - sample[mask][pred_sec] < rejection_criteria[1]:
                test_mask[[idx]] = False

        # Accepted Stats
        accepted_predictions = np.argmax(test_probabilities[test_mask], axis=1)
        self.accepted_accuracy = np.mean(accepted_predictions == self.datamanager.test_data[1][test_mask])

        # Rejected Stats
        inv_test_mask = np.logical_not(test_mask)
        self.rejection_rate = len(test_probabilities[inv_test_mask]) / len(test_predictions)
        rejected_predictions = np.argmax(test_probabilities[inv_test_mask], axis=1)
        self.rejected_accuracy = np.mean(rejected_predictions == self.datamanager.test_data[1][inv_test_mask])

        return None

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
                f"Learning rate: {self.config['scheduler']['params']['lr_max']} \
                \n{"─" * 15} Loss {"─" * 20} \
                \nTraining Loss:\t{train_results['training_losses'][-1]:.3} \
                \nValid Loss:\t{train_results['validation_losses'][-1]:.3} \
                \n{"─" * 15} Accuracies {"─" * 15} \
                \nTraining Acc.:\t{train_results['training_accuracies'][-1]:.3%} \
                \nValid Acc.:\t{train_results['validation_accuracies'][-1]:.3%} \
                \nTest Acc.:\t{test_accuracy:.3%}\n")

        if self.rejection_metrics:
            print(f"{"─" * 15} Rejection Strategy {"─" * 15} \
                \nAccepted Acc.:\t{self.accepted_accuracy:.3%} \
                \nRejected Acc.:\t{self.rejected_accuracy:.3%} \
                \nRejection Rate:\t{self.rejection_rate:.3%}\n")

        experiment_filepath = os.path.join(
            self.config['log_filepath'], 
            model_name
        )
        if not os.path.exists(experiment_filepath):
            os.makedirs(experiment_filepath)

        experiment_params = f"_e{self.config['epochs']}_b{self.config['dataset']['batch_size']}_lr{self.config['scheduler']['params']['lr_max']:.2}"
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
