import os
import yaml
from yacs.config import CfgNode

# YACS overwrite these settings using YAML configs
# all YAML variables MUST BE defined here first as this is the master list of ALL attributes.

def load_config(config_file: str):
    """Loads the experiment configuration from a YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return CfgNode(config)

def get_cfg_defaults():
    """Returns a YACS CfgNode with the default configuration."""
    _C = CfgNode()

    # Dataset
    _C.dataset = CfgNode()
    _C.dataset.batch_size = 64
    _C.dataset.encoder = "onehot"
    _C.dataset.shuffle_train_set = True
    _C.dataset.shuffle_test_set = True
    _C.dataset.validation_set_length = 10000
    _C.dataset.train_images_filepath = "./data/MNIST/train-images"
    _C.dataset.train_labels_filepath = "./data/MNIST/train-labels"
    _C.dataset.test_images_filepath = "./data/MNIST/test-images"
    _C.dataset.test_labels_filepath = "./data/MNIST/test-labels"

    _C.epochs = 2

    # Model and Layer Configuration
    _C.layers = []
    _C.layers.append(CfgNode())
    _C.layers[-1].name = "dense"
    _C.layers[-1].params = CfgNode()
    _C.layers[-1].params.input_size = 784
    _C.layers[-1].params.output_size = 1000
    _C.layers[-1].params.weight_init = "he"

    _C.layers.append(CfgNode())
    _C.layers[-1].name = "batchnorm"
    _C.layers[-1].params = CfgNode()
    _C.layers[-1].params.dim = 1000

    _C.layers.append(CfgNode())
    _C.layers[-1].name = "relu"

    _C.layers.append(CfgNode())
    _C.layers[-1].name = "dropout"
    _C.layers[-1].params = CfgNode()
    _C.layers[-1].params.rate = 0.4

    _C.layers.append(CfgNode())
    _C.layers[-1].name = "dense"
    _C.layers[-1].params = CfgNode()
    _C.layers[-1].params.input_size = 1000
    _C.layers[-1].params.output_size = 10
    _C.layers[-1].params.weight_init = "xavier"

    _C.layers.append(CfgNode())
    _C.layers[-1].name = "softmax"

    # Scheduler
    _C.scheduler = CfgNode()
    _C.scheduler.name = "warmup"
    _C.scheduler.params = CfgNode()
    _C.scheduler.params.base_scheduler = CfgNode()
    _C.scheduler.params.base_scheduler.name = "step"
    _C.scheduler.params.base_scheduler.params = CfgNode()
    _C.scheduler.params.base_scheduler.params.lr_start = 9e-2
    _C.scheduler.params.base_scheduler.params.decay_factor = 0.9
    _C.scheduler.params.base_scheduler.params.step_size = 200
    _C.scheduler.params.lr_start = 1e-3
    _C.scheduler.params.lr_max = 9e-2
    _C.scheduler.params.warmup_steps = 784

    # Logging
    _C.log_filepath = "./results/logs/"

    return _C.clone()