
import copy
from neuralnet import *

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
    print(config)
    config_str=str(config['layer_specs'])+'_'+str(config['activation'])+'_'+str(config['learning_rate'])+'_'+str(config['batch_size'])+'_'+str(config['L2_penalty'])+'_'+str(config['momentum_gamma'])


    corr_arr=[]
    loss_arr=[]
    val_loss=[]
    val_acc=[]
    best_loss_epoch = 0
    best_loss = np.inf
    epoch_increase_counter = 0
    print('working on epoch:')
    for epoch in range(config['epochs']):
        print(epoch)
        x_train, y_train = util.shuffle((x_train, y_train))
        epoch_corr_arr=[]
        epoch_loss_arr=[]
        miniGen = util.generate_minibatches((x_train,y_train),batch_size=config['batch_size'])
        for mini_batch in miniGen:
            x_batch=mini_batch[0]
            y_batch=mini_batch[1]
            corr, loss = model.forward(x_batch,y_batch)
            epoch_corr_arr.append(corr)
            epoch_loss_arr.append(loss)
            model.backward()
        corr_arr.append(np.mean(epoch_corr_arr))
        loss_arr.append(np.mean(epoch_loss_arr))

        val_acc_epoch, val_loss_epoch = model.forward(x_valid, y_valid)
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)
    
        if config['early_stop']:
            if val_loss_epoch < best_loss:
                best_loss = val_loss_epoch
                best_loss_epoch = epoch
                best_model = copy.deepcopy(model)
                epoch_increase_counter = 0
            elif val_loss_epoch > val_loss[-2]:
                epoch_increase_counter += 1
            else:
                epoch_increase_counter = 0
            if epoch_increase_counter >= config['early_stop_epoch']:
                break

    if not config['early_stop']:
        best_model = model

    util.plots(loss_arr, corr_arr, val_loss, val_acc, best_loss_epoch,config_str)

    # Read in the esssential configs

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
    acc,loss=model.forward(X_test,y_test)
    return acc, loss


