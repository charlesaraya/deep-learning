config = {
    'train_images_filepath': './data/MNIST/train-images',
    'train_labels_filepath': './data/MNIST/train-labels',
    'test_images_filepath': './data/MNIST/test-images',
    'test_labels_filepath': './data/MNIST/test-labels',
    'shuffle_train_set': True,
    'shuffle_test_set': False,
    'validation_set_length': 10000,
    'batch_size': 64,
    'input_layer': 784,
    'hidden_layers': [256],
    'output_layer': 10,
    'epochs': 20,
    'learning_rate': 1e-3,
    'log_filepath': './results/logs/'
}