import ast

from optimizers.sgd import SGD
from optimizers.rmsprop import RMSProp

OPTIMIZERS = {
    'sgd': SGD,
    'rmsprop': RMSProp,
}

class OptimizerFactory:
    """Implements a factory class for creating optimizers based on configuration.

    This class provides an interface for creating an optimizer dynamically by mapping 
    optimizer type to its respective class. It uses a configuration dictionary 
    that defines the optimizer type and its parameters. The factory ensures that 
    only valid optimizer types are used and can handle parameter parsing for different 
    optimizers.
    """
    def __init__(self):
        """Initializes the OptimizerFactory with a predefined optimizer map.

        The optimizer map is a dictionary containing the mapping between optimizer names 
        and their associated optimizer class constructors (e.g., convolutional, dense, etc.)
        """
        self.map = OPTIMIZERS

    def create(self, name: str, params: dict = {}):
        """Creates a optimizer based on the provided configuration.

        This method parses the configuration dictionary to extract the optimizer type and its parameters. 
        It validates the optimizer type and initializes the corresponding optimizer with the provided parameters.

        #### Args
            - `config` (`dict`): A dictionary containing the optimizer configuration. 
                - `'name'`: Specifies the optimizer type (e.g., `'sgd'`),
                - `'params'` (optional): Specifies the parameters for initializing the optimizer.

        #### Returns
            - `object`: The instance of the specified optimizer class initialized with the provided parameters.
        """
        self.params = params
        self.params['name'] = name

        if name not in self.map:
            raise ValueError(f'Unknown optimizer: {name}')
        return self.map[name](**params)