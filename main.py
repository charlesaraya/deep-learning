from src.experiments.experiment_runner import ExperimentRunner
from src.data.mnist_data import MNISTDatasetManager
from src.model.basemodel import BaseModel
from src.experiments.config import config

def main():
    experiment = ExperimentRunner(BaseModel, MNISTDatasetManager, config)
    experiment.run()

if __name__ == "__main__":
    main()