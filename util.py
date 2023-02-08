import copy
import os, gzip
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import constants


def load_config(path):
    """
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    Normalizes inputs (on per channel basis of every image) here to have 0 mean and unit variance.
    This will require reshaping to seprate the channels and then undoing it while returning

    args:
        inp : N X d 2D array where N is the number of examples and d is the number of dimensions

    returns:
        normalized inp: N X d 2D array

    """
    arr_3d = inp.reshape(inp.shape[0], 3, 32 * 32)
    mu = np.mean(arr_3d, axis=2).reshape(inp.shape[0], 3, 1)
    sigma = np.std(arr_3d, axis=2).reshape(inp.shape[0], 3, 1)
    arr_3d_normal = (arr_3d - mu) / sigma
    return arr_3d_normal.reshape(inp.shape[0], 32 * 32 * 3)


def one_hot_encoding(labels, num_classes=20):
    """
    Encodes labels using one hot encoding.

    args:
        labels : N dimensional 1D array where N is the number of examples
        num_classes: Number of distinct labels that we have (20/100 for CIFAR-100)

    returns:
        oneHot : N X num_classes 2D array

    """


    """count = 0
    label_dict = {}
    for label in labels:
        if label not in label_dict.keys():
            label_dict[label] = count
            count += 1
    num_labels = np.array([label_dict[label] for label in labels])"""

    return np.eye(num_classes)[labels.flatten()]


def generate_minibatches(dataset, batch_size=64):
    """
        Generates minibatches of the dataset

        args:
            dataset : 2D Array N (examples) X d (dimensions)
            batch_size: mini batch size. Default value=64

        yields:
            (X,y) tuple of size=batch_size

        """

    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def calculate_correct(y, t):
    # Feel free to use this function to return accuracy instead of number of correct predictions
    """
    Calculates the number of correct predictions

    args:
        y: Predicted Probabilities
        t: Labels in one hot encoding

    returns:
        the accuracy of prediction
    """
    y_pred = np.argmax(y, axis=1)
    t_pred = np.argmax(t, axis=1)
    return np.sum(y_pred == t_pred) / y.shape[0]


def append_bias(X):
    """
    Appends bias to the input
    args:
        X (N X d 2D Array)
    returns:
        X_bias (N X (d+1)) 2D Array
    """
    return np.insert(X, 0, 1, axis=1)


def plots(train_epoch_loss, train_epoch_accuracy, val_epoch_loss, val_epoch_accuracy, early_stop, name=''):
    """
    Helper function for creating the plots
    earlyStop is the epoch at which early stop occurred and will correspond to the best model. e.g. earlyStop=-1 means the last epoch was the best one
    """

    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1, len(train_epoch_loss) + 1, 1)
    ax1.plot(epochs, train_epoch_loss, 'r', label="Training Loss")
    ax1.plot(epochs, val_epoch_loss, 'g', label="Validation Loss")
    plt.scatter(epochs[early_stop], val_epoch_loss[early_stop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=35)
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(constants.saveLocation + "loss_" + name + ".png")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, train_epoch_accuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, val_epoch_accuracy, 'g', label="Validation Accuracy")
    plt.scatter(epochs[early_stop], val_epoch_accuracy[early_stop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('Accuracy Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('Accuracy', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    plt.savefig(constants.saveLocation + "accuracy_" + name + ".png")
    plt.show()

    # Saving the losses and accuracies for further offline use
    pd.DataFrame(train_epoch_loss).to_csv(constants.saveLocation + "trainEpochLoss_" + name + ".csv")
    pd.DataFrame(val_epoch_loss).to_csv(constants.saveLocation + "valEpochLoss_" + name + ".csv")
    pd.DataFrame(train_epoch_accuracy).to_csv(constants.saveLocation + "trainEpochAccuracy_" + name + ".csv")
    pd.DataFrame(val_epoch_accuracy).to_csv(constants.saveLocation + "valEpochAccuracy_" + name + ".csv")


def shuffle(dataset):
    """
    Shuffle dataset.

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    sigma = np.random.permutation(dataset[0].shape[0])

    X_shuffled = dataset[0][sigma]
    y_shuffled = dataset[1][sigma]

    return X_shuffled, y_shuffled


def create_train_val_split(x_train, y_train):
    """
    Creates the train-validation split (80-20 split for train-val). Please shuffle the data before creating the train-val split.
    """
    train_amount = np.ceil(.8 * x_train.shape[0]).astype(int)

    X_shuffled, y_shuffled = shuffle((x_train, y_train))

    return X_shuffled[:train_amount, :], y_shuffled[:train_amount], X_shuffled[train_amount:, :], y_shuffled[
                                                                                                  train_amount:]


def load_data(path):
    """
    Loads, splits our dataset- CIFAR-100 into train, val and test sets and normalizes them

    args:
        path: Path to cifar-100 dataset
    returns:
        train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels

    """

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, constants.cifar100_directory)

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    images_dict = unpickle(os.path.join(cifar_path, "train"))
    data = images_dict[b'data']
    label = images_dict[b'coarse_labels']
    train_labels.extend(label)
    train_images.extend(data)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels) #.reshape((len(train_labels), -1))
    train_images, train_labels, val_images, val_labels = create_train_val_split(train_images, train_labels)

    train_normalized_images = normalize_data(train_images)
    train_one_hot_labels = one_hot_encoding(train_labels, num_classes=20)

    val_normalized_images = normalize_data(val_images)
    val_one_hot_labels = one_hot_encoding(val_labels, num_classes=20)

    test_images_dict = unpickle(os.path.join(cifar_path, "test"))
    test_data = test_images_dict[b'data']
    test_labels = test_images_dict[b'coarse_labels']
    test_images = np.array(test_data)
    test_labels = np.array(test_labels).reshape((len(test_labels), -1))
    test_normalized_images = normalize_data(test_images)
    test_one_hot_labels = one_hot_encoding(test_labels, num_classes=20)
    return train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels, \
           test_normalized_images, test_one_hot_labels

def load_data_fine(path):
    """
    Loads, splits our dataset- CIFAR-100 into train, val and test sets and normalizes them
    Uses fine labels instead of coarse

    args:
        path: Path to cifar-100 dataset
    returns:
        train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels

    """

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, constants.cifar100_directory)

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    images_dict = unpickle(os.path.join(cifar_path, "train"))
    data = images_dict[b'data']
    label = images_dict[b'fine_labels']
    train_labels.extend(label)
    train_images.extend(data)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels) #.reshape((len(train_labels), -1))
    train_images, train_labels, val_images, val_labels = create_train_val_split(train_images, train_labels)

    train_normalized_images = normalize_data(train_images)
    train_one_hot_labels = one_hot_encoding(train_labels, num_classes=100)

    val_normalized_images = normalize_data(val_images)
    val_one_hot_labels = one_hot_encoding(val_labels, num_classes=100)

    test_images_dict = unpickle(os.path.join(cifar_path, "test"))
    test_data = test_images_dict[b'data']
    test_labels = test_images_dict[b'fine_labels']
    test_images = np.array(test_data)
    test_labels = np.array(test_labels).reshape((len(test_labels), -1))
    test_normalized_images = normalize_data(test_images)
    test_one_hot_labels = one_hot_encoding(test_labels, num_classes=100)
    return train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels, \
           test_normalized_images, test_one_hot_labels
