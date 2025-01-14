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
            'input': 784,
            'output': 1000,
            'weight_init': 'he'
        },
        {
            'name': 'batchnorm',
            'dim': 1000
        },
        {
            'name': 'relu'
        },
        {
            'name': 'dropout',
            'rate': 0.4
        },
        {
            'name': 'dense',
            'input': 1000,
            'output': 10,
            'weight_init': 'xavier'
        },
        {
            'name': 'softmax'
        }
    ],
    'scheduler': {
        'main': {
            'name': 'warmup',
            'learning_rate_start': 1e-3,
            'learning_rate': 9e-2,
            'warmup_ratio': 0.1
        },
        'base': {
            'name': 'step',
            'learning_rate': 9e-2,
            'decay_factor': 0.9,
            'step_ratio': 0.15
        },
    },
    'log_filepath': './results/logs/'
}