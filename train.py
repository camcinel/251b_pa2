
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
    best_loss_epoch = 0
    best_loss = np.inf
    epoch_increase_counter = 0

    for epoch in tqdm(range(config['epochs']), desc='epochs', total=float('inf'), leave=False):
        x_train, y_train = util.shuffle((x_train, y_train))
        epoch_loss = []
        epoch_acc = []
        for mini_batch in generate_minibatches((x_train, y_train), batch_size=config['batch_size']):
            acc, loss = model.forward(mini_batch[0], mini_batch[1])
            epoch_loss.append(loss)
            epoch_acc.append(acc)
            model.backward()
        train_loss.append(np.mean(epoch_loss))
        train_acc.append(np.mean(epoch_acc))

        val_acc_epoch, val_loss_epoch = model.forward(x_valid, y_valid)
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)

        if config['early_stop']:
            if val_loss_epoch < best_loss:
                best_loss = val_loss_epoch
                best_loss_epoch = epoch
                best_model = copy.deepcopy(model)
                epoch_increase_counter = 0
            else:
                epoch_increase_counter += 1

            if epoch_increase_counter >= config['early_stop_epoch']:
                print(f'\n Early stop at epoch {epoch}')
                break

    if not config['early_stop']:
        best_model = model

    util.plots(train_loss, train_acc, val_loss, val_acc, best_loss_epoch, name=config['name'])

    return best_model

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


