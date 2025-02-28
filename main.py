import os

from src.experiments.experiment_runner import ExperimentRunner
from src.data.mnist_data import MNISTDatasetManager
from model.model import Model
from src.experiments.config import load_config, get_cfg_defaults
from utils.global_logger import Logger

def main():

    logger = Logger()
    logger = logger.setup_daily_logger(console=True)
    logger.info("Starting program...")

    # Load default configuration
    config = get_cfg_defaults()

    for file_name in os.listdir(config.config_dir):
        if not file_name.endswith(('.yaml')):
            continue

        # Merge default config with experiment config
        file_path = os.path.join(config.config_dir, file_name)
        experiment_config = load_config(file_path)
        if experiment_config['run_experiment']:
            config.merge_from_other_cfg(experiment_config)
            # Prep and run experiment
            logger.info(f"Starting Experiment: {file_name}, {config['model']['name']}")
            experiment = ExperimentRunner(Model, MNISTDatasetManager, config)
            logger.info(f"Running Experiment: {file_name}, {config['model']['name']}")
            experiment.run()

    logger.info(f"Program finalized.")

if __name__ == "__main__":
    main()