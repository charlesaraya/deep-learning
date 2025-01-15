from layers.denselayer import DenseLayer
from layers.batchnorm import BatchNorm
from layers.activations import *
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

class LayerFactory:
    def __init__(self):
        self.layer_map = LAYERS

    def create(self, layer_config):
        layer_type = layer_config['name']
        if layer_type not in self.layer_map:
            raise ValueError(f'Unknown layer type: {layer_type}')
        return self.layer_map[layer_type](**layer_config.get('params', {}))