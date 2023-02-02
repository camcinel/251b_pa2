import numpy as np
from neuralnet import Neuralnetwork

def check_grad(model, x_train, y_train):

    """
    TODO
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    epsilon = 1e-2

    correct,loss=model.forward(x_train,targets=y_train)
    print("correct:", correct, " loss: ", loss)

    correctP,lossP,correctM,lossM=model.forwardEpsilon(x_train,epsilon,1,0,targets=y_train)
    print("loss P:", lossP, " loss M: ", lossM)

    print("numerical:",(lossP-lossM)/(2*epsilon))
    numerical_dEdw=(lossP-lossM)/(2*epsilon)

    correct,loss=model.forward(x_train,targets=y_train)
    model.backward()
    backprop_dEdw=-1*model.layers[1].dw[0][8]
    print("backprop:",backprop_dEdw)

    print('diff:',numerical_dEdw-backprop_dEdw)

    #model.printLayerStructure()
    #correct,loss=model.forward(x_train,targets=y_train)
    #print("correct:", correct, " loss: ", loss)
    #model.backward()
    #model.printLayerStructure()


def checkGradient(x_train,y_train,config):

    subsetSize = 5 #Feel free to change this
    sample_idx = np.random.randint(0,len(x_train),subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)
