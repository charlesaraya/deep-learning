# Deep Learning from Scrath
This repository is a continuous learning project that serves as playground for exploring deep learning's most fundamental and advanced concepts, where I implement its core components from scratch, without relying on high-level libraries like TensorFlow or PyTorch.

The goal is to deepen my understanding of how deep learning works under the hood while showcasing a modular and scalable implemenation of neural network layers, datasets, training and evaluation features to experiments with.

## Feature Evolution
- **MLP model**:
  - Weight initialization: random, glorot, and he.
  - One-hot encoder.
  - Forward and backpropagation.
  - Online and Mini-batch learning.
  - Sigmoid, Tanh, ReLU, and Softmax activations.
  - Cross-entropy and MSE loss functions.
  - Metric monitoring: training & validation loss and accuracy.
  - MNIST datamanager: loads raw data set, implements an iterator for mini-batch learning, preps data, and has data visualization support.
  - Dropout regarization.
  - Batch Normazation.
  - Label Smoothing.
  - Learning schedulers, with step decay, exponential decay, cosine annealing decay schedulers.
  - Data augmentation: rotation, translation, scaling, shearing, and noising.
- **Model and Data manager enhancements**:
  - Modular base model with sequential layers.
  - Experiment runner: creates and streams experiments set-up via configuration.
  - Activation and Loss modules.
  - Layer interface, Dense layer, and Layer factory.
  - Scheduler factory.
  - Checkpoints, save and load.
- **Convolutional Neural Networks**.
  - Convolution layer.
  - Pooling layer, with max and average pooling support.
  - Flattening layer.
  - EMNIST support.
- **Model enhancements**:
  - Rejection-based classification support.
  - Update method, where each layer performs gradient descent.
  - Evaluate method, with mini-batch support.
  - Optimizer module, and SGD with momentum optimizer.
  - Compile method, to set-up the optimizer and loss function.
  - Summary method, to print layer and parameter summary.