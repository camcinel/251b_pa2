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
    epsilon = 1.11e-2

    correct,loss=model.forward(x_train,targets=y_train)
    print("correct:", correct, " loss: ", loss)

    correctP,lossP,correctM,lossM=model.forwardEpsilon(x_train,epsilon,1,0,1,targets=y_train)
    print("loss P:", lossP, " loss M: ", lossM)

    print("numerical:",(lossP-lossM)/(2*epsilon))
    numerical_dEdw=(lossP-lossM)/(2*epsilon)

    correct,loss=model.forward(x_train,targets=y_train)
    model.backward()
    backprop_dEdw=-1*model.layers[1].dw[0][1]
    print("backprop:",backprop_dEdw)

    print('diff:',numerical_dEdw-backprop_dEdw)

    """
    Data Log: 
    1 EXAMPLE SUBSET

    FIRST LAYER BIAS (1)
        layeridx=0 , biasORhidden=0, epsilon = 1.11e-2 , weightdir_idx = 3
        correct: 0  loss:  2.999707874849857
        loss P: 2.9997577488319846  loss M:  2.9996581928870083
        numerical: 0.004484502025961637
        backprop:  0.004484669519803634
        diff: -1.6749384199737927e-07

    SECOND LAYER BIAS (1)
        1 , 0 , 3
        correct: 0  loss:  3.003876008799854
        loss P: 3.0044396038622936  loss M:  3.003318322476271
        numerical: 0.05050817054155001
        backprop:  0.05050728523402751
        diff: 8.85307522499601e-07

    FIRST LAYER HIDDEN (2)
        0, 45, 17
        correct: 0  loss:  3.000230353110124
        loss P: 3.000228074169166  loss M:  3.000232632609933
        numerical: -0.00020533516968816858
        backprop: -0.0002053351949206263
        diff: 2.5232457724795598e-11

        0 ,11, 4
        correct: 0  loss:  3.0019029434657907
        loss P: 3.0019136979913346  loss M:  3.0018921758233117
        numerical: 0.0009694670280571818
        backprop:  0.000969474480757086
        diff: -7.4526999041492e-09


    SECOND LAYER HIDDEN (2)
        1, 11, 4
        correct: 1  loss:  2.9854492649156463
        loss P: 2.9856676495868353  loss M:  2.98523178867533
        numerical: 0.01963337439212429
        backprop:  0.01963332050665732
        diff: 5.3885466970438056e-08

        1, 45 , 17
        correct: 0  loss:  2.996272748678935
        loss P: 2.996294808516214  loss M:  2.996250698009269
        numerical: 0.001986959772311043
        backprop:  0.001986959718140762
        diff: 5.417028110799027e-11

    5 EXAMPLE SUBSET
        1, 12, 12
        correct: 0  loss:  4.607038986333934
        loss P: 4.607052383845708  loss M:  4.6070257222739155
        numerical: 0.0012009717023467139
        backprop: 0.0012009700144157034
        diff: 1.6879310104581796e-09

        0,300,12
        correct: 0  loss:  4.606198531155811
        loss P: 4.606220075512859  loss M:  4.60617716984697
        numerical: 0.00193268765263982
        backprop: -0.003879678807450254
        diff: 0.005812366460090074

        0 , 0 , 103
        correct: 0  loss:  4.605389979984842
        loss P: 4.605396659724635  loss M:  4.605383242054959
        numerical: 0.0006043995349448129
        backprop: -0.01724593296663597
        diff: 0.017850332501580784

        1, 0, 19
        correct: 0  loss:  4.608938485465016
        loss P: 4.609494866750987  loss M:  4.608387941240354
        numerical: 0.049861509487996225
        backprop: 0.04986063366172454
        diff: 8.758262716873633e-07

    """


def checkGradient(x_train,y_train,config):

    subsetSize = 33 #Feel free to change this
    sample_idx = np.random.randint(0,len(x_train),subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)
