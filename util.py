import copy
import os, gzip
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import constants

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
    X,y=dataset
    l=len(X)
    index_list=np.arange(l)
    np.random.shuffle(index_list)
    images=np.array([X[i] for i in index_list])
    labels=np.array([y[i] for i in index_list])
    return images, labels

def z_score_normalize(X, u = None, sd = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / sigma
        where 
            μ = mean of x
            sigma = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    u=np.mean(X)
    sd=np.std(X)
    newX=(X-u)/sd
    return newX

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
    returnarr=np.zeros([len(inp),len(inp[0])])
    L=len(inp[0])//3
    for i in range(len(inp)):
        oneimage=np.zeros(len(inp[i])//3)
        for c in range(3):
            oneimage=z_score_normalize(inp[i][c*L:(c+1)*L])
            returnarr[i][c*L:(c+1)*L]=oneimage
    return returnarr
    #raise NotImplementedError("normalize_data not implemented")



def one_hot_encoding(labels, num_classes=20):
    """
    Encodes labels using one hot encoding.

    args:
        labels : N dimensional 1D array where N is the number of examples
        num_classes: Number of distinct labels that we have (20/100 for CIFAR-100)

    returns:
        oneHot : N X num_classes 2D array

    """
    targets=np.arange(0,num_classes,1)
    k=len(labels)
    onehotarr=np.zeros([k,num_classes])
    for i in range(k):
        for j in range(num_classes):
            onehotarr[i][j]=(labels[i]==targets[j])
    return onehotarr
    #raise NotImplementedError("one_hot_encoding not implemented")



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


def calculateCorrect(y,t):  #Feel free to use this function to return accuracy instead of number of correct predictions
    """
    TODO
    Calculates the number of correct predictions

    args:
        y: Predicted Probabilities
        t: Labels in one hot encoding

    returns:
        the number of correct predictions
    """
    return np.sum([(np.argmax(y[i])==np.argmax(t[i])) for i in range(len(y))])
    #raise NotImplementedError("calculateCorrect not implemented")



def append_bias(X):
    """
    Appends bias to the input
    args:
        X (N X d 2D Array)
    returns:
        X_bias (N X (d+1)) 2D Array
    """
    return np.insert(X,0,1,axis=1)

    #raise NotImplementedError("append_bias not implemented")




def plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop):

    """
    Helper function for creating the plots
    earlyStop is the epoch at which early stop occurred and will correspond to the best model. e.g. earlyStop=-1 means the last epoch was the best one
    """

    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1,len(trainEpochLoss)+1,1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'g', label="Validation Loss")
    plt.scatter(epochs[earlyStop],valEpochLoss[earlyStop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35 )
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(constants.saveLocation+"loss.eps")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, trainEpochAccuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, 'g', label="Validation Accuracy")
    plt.scatter(epochs[earlyStop], valEpochAccuracy[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('Accuracy Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('Accuracy', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    plt.savefig(constants.saveLocation+"accuarcy.eps")
    plt.show()

    #Saving the losses and accuracies for further offline use
    pd.DataFrame(trainEpochLoss).to_csv(constants.saveLocation+"trainEpochLoss.csv")
    pd.DataFrame(valEpochLoss).to_csv(constants.saveLocation+"valEpochLoss.csv")
    pd.DataFrame(trainEpochAccuracy).to_csv(constants.saveLocation+"trainEpochAccuracy.csv")
    pd.DataFrame(valEpochAccuracy).to_csv(constants.saveLocation+"valEpochAccuracy.csv")



def createTrainValSplit(x_train,y_train):

    """
    Creates the train-validation split (80-20 split for train-val).
    Please shuffle the data before creating the train-val split.
    """
    shuf_x, shuf_y = shuffle((x_train,y_train))
    l=len(y_train)
    spt_idx = int((l//5)*4)
    train_im = shuf_x[:spt_idx]
    train_lab= shuf_y[:spt_idx]
    val_im = shuf_x[spt_idx:]
    val_lab = shuf_y[spt_idx:]
    return train_im, train_lab, val_im, val_lab
    #raise NotImplementedError("createTrainValSplit not implemented")



def load_data(path,dataSize):
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
    train_labels = np.array(train_labels).reshape((len(train_labels),-1))
    train_images, train_labels, val_images, val_labels = createTrainValSplit(train_images,train_labels)
    if dataSize==0:
        train_images=train_images[:100]
        train_labels=train_labels[:100]
        val_images= val_images[:20]
        val_labels= val_labels[:20]
    print('normalizing train images ....')
    train_normalized_images =  normalize_data(train_images) 
    train_one_hot_labels = one_hot_encoding(train_labels) 

    print('normalizing validation images ....')
    val_normalized_images = normalize_data(val_images) 
    val_one_hot_labels = one_hot_encoding(val_labels) 

    test_images_dict = unpickle(os.path.join(cifar_path, "test"))
    test_data = test_images_dict[b'data']
    test_labels = test_images_dict[b'coarse_labels']
    test_images = np.array(test_data)
    test_labels = np.array(test_labels).reshape((len(test_labels), -1))
    if dataSize==0:
        test_images=test_images[:20]
        test_labels=test_labels[:20]
    test_normalized_images= normalize_data(test_images) 
    test_one_hot_labels = one_hot_encoding(test_labels)

    return train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels


def plotImage(X):
    print(type(X))
    L=len(X)//3
    R=X[:L]
    G=X[L:2*L]
    B=X[2*L:3*L]
    print(type(B))
    BW=np.array([np.mean([R[i],G[i],B[i]]) for i in range(L)])

    fig, ax = plt.subplots(2,2)
    ax[0][0].pcolor(R.reshape(32,32),cmap='Reds',vmin=-2,vmax=2)
    ax[0][1].pcolor(G.reshape(32,32),cmap='Greens',vmin=-2,vmax=2)
    ax[1][0].pcolor(B.reshape(32,32),cmap='Blues',vmin=-2,vmax=2)
    ax[1][1].pcolor(BW.reshape(32,32),cmap='Greys',vmin=-2,vmax=2)
    ax[0][0].set_ylim([32,0])
    ax[0][1].set_ylim([32,0])
    ax[1][0].set_ylim([32,0])
    ax[1][1].set_ylim([32,0])

    plt.show()
    plt.close()
    plt.clf
