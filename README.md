# Image Classification for CIFAR-100

## Description

This python module trains a simple network for classification of the [CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
Currently, it can perform classification on the 20 coarse and 100 fine labels.
There is also functionality to compare gradients calculated through numerical approximations to those calculated through backpropagation.

For all classification trainings, the module will create charts of average loss and accuracy over each training epoch.
These charts are outputted in the created `/plots` directory, along with `.csv` files with the actual values.

## Usage

To use the module, first the data must be loaded via the shell script
```
sh get_cifar100data.sh
```

Then training experiments can be run via
```
python main.py
```

### Command Line Arguments

This module supports the command line argument `--experiment` with the following allowed values:

- `test_gradient`: shows comparison between numerically approximated gradients and those obtained through backpropagation
- `test_momentum` (default): trains a network on coarse labels with tanh activation, 128 hidden units, and momentum coefficient of 0.9
- `test_regularization`: trains a network with the same configuration as previous experiment, along with 10^(-5) L2 penalty coefficient
- `test_activation`: trains a network with the same configuration as the previous experiment, except with ReLU activation function
- `test_half_units`: trains a network with the same configuration as the previous experiment, except the hidden units are halved to 64 units
- `test_double_units`: trains a network with the same configuration as `test_activation`, except the number of hidden units is doubled to 256 hidden units
- `test_hidden_layers`: trains a network with the same configuration as `test_activation`, except it uses two hidden layers of 123 units each
- `test_100_classes`: trains a network with the same configuration as `test_activation` on the fine labels

## Required Libraries

The following libraries are required to run the module:

- `numpy`: for matrix calculations
- `matplotlib`: for creating loss and accuracy charts
- `pandas`: to export `.csv` files for losses and accuracies across training
- `pyyaml`: for reading the config files

## File Structure

The module is broken up into the following files:
- `get_cifar100data.sh`: shell script to download the CIFAR dataset
- `constants.py`: contains constants such as data directory and figure output directory
- `gradient.py`: contains functions to test the differently calculated gradients
- `util.py`: contains helper functions for loading and processing data and creating charts
- `train.py`: contains functions to train and test models
- `neuralnet.py`: contains the classes to create the neural network
- `main.py`: primary script. Parsers command line arguments and run the experiments