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
    epsilon = 1.01e-2
    layer_idx = 1 
    weight_idx1 = 46 # 0 means bias and any other number works too
    weight_idx2 = 2 # the index of the output on the next layer

    correct,loss=model.forward(x_train,targets=y_train)
    #print("correct:", correct, " loss: ", loss)

    correctP,lossP,correctM,lossM=model.forwardEpsilon(x_train,epsilon,layer_idx,weight_idx1,weight_idx2,targets=y_train)
    print("loss P:", lossP, " loss M: ", lossM)

    print('epsilon:',epsilon,', layer:',layer_idx,', wt_1:',weight_idx1,', wt_2:',weight_idx2)
    print("numerical:",(lossP-lossM)/(2*epsilon))
    numerical_dEdw=(lossP-lossM)/(2*epsilon)

    correct,loss=model.forward(x_train,targets=y_train)
    model.backward(gradReqd=False)
    backprop_dEdw=-1*model.layers[layer_idx].dw[weight_idx1][weight_idx2]/model.learning_rate
    print("backprop:",backprop_dEdw)

    print('diff:',numerical_dEdw-backprop_dEdw)



def checkGradient(x_train,y_train,config):

    subsetSize = 5 #Feel free to change this
    sample_idx = np.random.randint(0,len(x_train),subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)
    
"""
Data Log FINAL

FIRST LAYER HIDDEN (2)
epsilon: 0.0101 , layer: 0 , wt_1: 123 , wt_2: 100
numerical: -0.0014127850181482613
backprop: -0.0014127890825363205
diff: 4.064388059183549e-09

epsilon: 0.0101 , layer: 0 , wt_1: 2023 , wt_2: 91
numerical: -0.000931633700921715
backprop: -0.0009316378032827894
diff: 4.1023610743238376e-09

SECOND LAYER HIDDEN (2)

epsilon: 0.0101 , layer: 1 , wt_1: 104 , wt_2: 7
numerical: -0.009212998996087091
backprop: -0.009212994422717866
diff: -4.57336922499163e-09

epsilon: 0.0101 , layer: 1 , wt_1: 46 , wt_2: 2
numerical: 0.013887529533502412
backprop: 0.013887514081372242
diff: 1.5452130169152367e-08

FIRST LAYER BIAS (1)
epsilon: 0.0101 , layer: 0 , wt_1: 0 , wt_2: 81
numerical: 0.0016689142737087701
backprop: 0.0016689681657416774
diff: -5.3892032907251405e-08

SECOND LAYER BIAS (1)
epsilon: 0.0101 , layer: 1 , wt_1: 0 , wt_2: 8
numerical: 0.05001209103200179
backprop: 0.05001136407102267
diff: 7.26960979122171e-07

"""
