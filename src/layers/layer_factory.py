import ast

from layers.denselayer import DenseLayer
from layers.convlayer import ConvLayer
from layers.flatten import Flatten
from layers.batchnorm import BatchNorm
from layers.activations import *
from layers.pooling import Pooling
from layers.regularizations import Dropout

LAYERS = {
    'dense': DenseLayer,
    'conv2D': ConvLayer,
    'batchnorm': BatchNorm,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'relu': ReLU,
    'softmax': SoftMax,
    'pooling': Pooling,
    'dropout': Dropout,
    'flatten': Flatten
}

class LayerFactory:
    def __init__(self):
        self.layer_map = LAYERS

    def create(self, layer_config):
        layer_type = layer_config['name']
        if layer_type not in self.layer_map:
            raise ValueError(f'Unknown layer type: {layer_type}')
        # YACS loads yaml tuples as strings
        if 'params' in layer_config and 'input_shape' in layer_config['params']:
            layer_config['params']['input_shape'] = ast.literal_eval(layer_config['params']['input_shape'])
        return self.layer_map[layer_type](**layer_config.get('params', {}))