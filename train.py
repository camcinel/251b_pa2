
import copy
from neuralnet import *
from tqdm import tqdm
from util import generate_minibatches

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """

    # Read in the esssential configs
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for epoch in tqdm(range(100)):
        x_train, y_train = util.shuffle((x_train, y_train))
        epoch_loss = []
        epoch_acc = []
        for mini_batch in generate_minibatches((x_train, y_train)):
            acc, loss = model.forward(mini_batch[0], mini_batch[1])
            epoch_loss.append(loss)
            epoch_acc.append(acc)
            model.backward()
        train_loss.append(sum(epoch_loss) / len(epoch_loss))
        train_acc.append(sum(epoch_acc) / len(epoch_acc))
        val_acc_epoch, val_loss_epoch = model.forward(x_valid, y_valid)
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)

    util.plots(train_loss, train_acc, val_loss, val_acc, -1)

    return model

#This is the test method
def modelTest(model, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """
    return model.forward(X_test, y_test)


