import os

from src.experiments.experiment_runner import ExperimentRunner
from src.data.mnist_data import MNISTDatasetManager
from src.model.basemodel import BaseModel
from src.experiments.config import load_config, get_cfg_defaults

def main():
    # Load default configuration
    config = get_cfg_defaults()

    for file_name in os.listdir(config.config_dir):
        if not file_name.endswith(('.yaml')):
            raise Exception(f"File with wrong extension: {file_name}. Only yaml files are accepted.")

        # Merge default config with experiment config
        file_path = os.path.join(config.config_dir, file_name)
        experiment_config = load_config(file_path)
        config.merge_from_other_cfg(experiment_config)

        # Prep and run experiment
        experiment = ExperimentRunner(BaseModel, MNISTDatasetManager, config)
        experiment.run()

if __name__ == "__main__":
    main()