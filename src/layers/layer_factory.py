import ast

from layers.dense import Dense
from layers.conv import Conv
from layers.flatten import Flatten
from layers.batchnorm import BatchNorm
from layers.activations import *
from layers.pooling import Pooling
from layers.dropout import Dropout

LAYERS = {
    'dense': Dense,
    'conv2D': Conv,
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
    """Implements a factory class for creating layers based on configuration.

    This class provides an interface for creating layers dynamically by mapping 
    layer types to their respective classes. It uses a configuration dictionary 
    that defines the layer type and its parameters. The factory ensures that 
    only valid layer types are used and can handle parameter parsing for different 
    layers.
    """
    def __init__(self):
        """Initializes the LayerFactory with a predefined layer map.

        The layer map is a dictionary containing the mapping between layer names 
        and their associated layer class constructors (e.g., convolutional, dense, etc.)
        """
        self.layer_map = LAYERS
        self.production = {}

    def create(self, layer_config):
        """Creates a layer based on the provided configuration.

        This method parses the configuration dictionary to extract the layer type and its parameters. 
        It validates the layer type and initializes the corresponding layer with the provided parameters.

        #### Args
            - `layer_config` (`dict`): A dictionary containing the layer configuration. 
                - `'name'`: Specifies the layer type (e.g., `'dense'`, `'sigmoid'`, `'dropout'`),
                - `'params'` (optional): Specifies the parameters for initializing the layer.

        #### Returns
            - `object`: The instance of the specified layer class initialized with the provided parameters.
        """
        layer_name = layer_config['name']

        if layer_name not in self.production.keys():
            self.production[layer_name] = 1
        else:
            self.production[layer_name] += 1

        layer_params = layer_config.get('params', {}).copy()
        layer_params['uid'] = self.production[layer_name]
        if 'name' not in layer_params:
            layer_params['name'] = layer_name

        if layer_name not in self.layer_map:
            raise ValueError(f'Unknown layer type: {layer_name}')
        # YACS loads yaml tuples as strings
        if 'params' in layer_config and 'shape' in layer_config['params']:
            layer_params['shape'] = ast.literal_eval(layer_config['params']['shape'])
        return self.layer_map[layer_name](**layer_params)