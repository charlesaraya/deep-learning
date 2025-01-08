import numpy as np
from src.model.mlp import MLP
from src.data.mnist_data import MNISTDatasetManager
import json
import os

class ExperimentRunner(object):
    def __init__(self, model: MLP, datamanager: MNISTDatasetManager, config: dict):
        self.model = model
        self.datamanager = datamanager
        self.config = config

    def run(self) -> None:
        """Runs an experiment for a given configuration."""
        # Init Data Manager
        datamanager = self.datamanager(self.config['batch_size'])
        # Load Datasets
        datamanager.load_data(
            self.config['train_images_filepath'],
            self.config['train_labels_filepath'],
            type = 'train'
            )
        datamanager.load_data(
            self.config['test_images_filepath'],
            self.config['test_labels_filepath'],
            type = 'test'
            )
        # Prep Data
        datamanager.prepdata(
            type = 'train',
            shuffle = self.config['shuffle_train_set'],
            validation_len = self.config['validation_set_length']
        )
        test_data = datamanager.prepdata(
            type = 'test',
            shuffle = self.config['shuffle_test_set']
        )

        # Init Model
        model = self.model(
            self.config['input_layer'],
            self.config['hidden_layers'],
            self.config['output_layer']
        )
        # Train Model
        results = model.train(
            datamanager,
            self.config['epochs'],
            self.config['learning_rate']
        )
        # Evaluate
        test_accuracy = self.evaluate(model, test_data)

        # Log Results
        self.log_results(results, test_accuracy)

    def evaluate(self, model, test_data):
        """Evaluates the model on the test dataset."""
        # Inference
        test_probabilities = model.forward(test_data[0])
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
        model_name = self._create_model_name()
        print(f"\n{model_name}, epochs: {self.config['epochs']}, batch size: {self.config['batch_size']}, " +
                f"learning rate: {self.config['learning_rate']} \
                \nTraining Loss:\t{train_results['training_losses'][-1]:.3} \
                \nTraining Acc.:\t{train_results['training_accuracies'][-1]:.3%} \
                \nTest Acc.:\t{test_accuracy:.3%}\n")
        
        experiment_filepath = os.path.join(
            self.config['log_filepath'], 
            model_name
        )
        if not os.path.exists(experiment_filepath):
            os.makedirs(experiment_filepath)

        experiment_params = f"_e{self.config['epochs']}_b{self.config['batch_size']}_lr{self.config['learning_rate']:.2}"
        experiment_name = 'experiment_' + experiment_params + '.json'
        experiment_filepath = os.path.join(
            experiment_filepath,
            experiment_name
        )
        with open(experiment_filepath, 'w') as f:
            json.dump({
                'training_losses': train_results['training_losses'],
                'training_accuracies': train_results['training_accuracies']
                }, f)
