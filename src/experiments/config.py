config = {
    'dataset': {
        'train_images_filepath': './data/MNIST/train-images',
        'train_labels_filepath': './data/MNIST/train-labels',
        'test_images_filepath': './data/MNIST/test-images',
        'test_labels_filepath': './data/MNIST/test-labels',
        'shuffle_train_set': True,
        'shuffle_test_set': False,
        'validation_set_length': 10000,
        'batch_size': 64,
        'encoder': 'smoothlabel'
    },
    'epochs': 10,
    'layers': [
        {
            'name': 'dense',
            'params':{
                'input_size': 784,
                'output_size': 1000,
                'weight_init': 'he'
            }
        },
        {
            'name': 'batchnorm',
            'params': {
                'dim': 1000
            }
        },
        {
            'name': 'relu'
        },
        {
            'name': 'dropout',
            'params': {
                'rate': 0.4
            }
        },
        {
            'name': 'dense',
            'params': {
                'input_size': 1000,
                'output_size': 10,
                'weight_init': 'xavier'
            }
        },
        {
            'name': 'softmax'
        }
    ],
    'scheduler': {
        'name': 'warmup',
        'params': {
            'base_scheduler': {
                'name': 'step',
                'params': {
                    'lr_start': 9e-2,
                    'decay_factor': 0.9,
                    'step_size': 200
                }
            },
            'lr_start': 1e-3,
            'lr_max': 9e-2,
            'warmup_steps': 784
        }
    },
    'log_filepath': './results/logs/'
}