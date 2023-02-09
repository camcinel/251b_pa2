import numpy as np
import matplotlib.pyplot as plt
import util


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU",
                                   "output"]:  # output can be used for the final layer. Feel free to use/remove it
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

    def __call__(self, z):
        """
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)

    def sigmoid(self, x):
        """
        f(x) = 1 / (1 + exp(-x))
        """
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        f(x) = max(0, x)
        """
        return np.maximum(0, x)

    def output(self, x):
        """
        Sigmoid function:
            f(x)_i = exp(x_i) / sum(exp(x_j))
        """
        return np.exp(x - np.max(x, axis=1).reshape(-1, 1)) / np.sum(np.exp(x - np.max(x, axis=1).reshape(-1, 1)), axis=1)[:, None]

    def grad_sigmoid(self, x):
        """
        f(x) = exp(-x) / ((1 + exp(-x))^2)
        """
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    def grad_tanh(self, x):
        ""
        f(x) = 1 - tanh(x)^2
        """
        return 1 - np.tanh(x) ** 2

    def grad_ReLU(self, x):
        """
        f(x) = 1 if x > 0 else 0
        """
        return (x > 0) * 1

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """
        return 1


class Layer:
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation, weight_type):
        """
        Creates architecture for the layer

        args:
            in_units: dimension of input vectors
            out_units: dimension of output vectors
            activation: activation function for the layer
            weight_type: type of weights. Currently, only randomly selected weights are used
        """
        self.w = None
        if weight_type == 'random':
            self.w = 0.01 * np.random.random((in_units + 1, out_units))

        self.activation = activation  # Activation function

        self.x = None  # Save the input to forward in this
        self.a = None  # output without activation
        self.z = None  # Output After Activation


        self.dw = 0  # Save the gradient w.r.t w in this. You can have bias in w itself or uncomment the next line and handle it separately
        # self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Computes forward pass of layer

        args:
            x - input matrix of data

        returns:
            activate weighted input for the layer
        """
        self.x = util.append_bias(x)
        self.a = self.x @ self.w
        self.z = self.activation.forward(self.a)
        return self.z

    def backward(self, deltaCurrent, learning_rate, momentum_gamma, regularization, gradReqd=True):
        """
        Performs backwards pass for the layer

        args:
            deltaCur - weighted sum of deltas from the next layer
            learning_rate - multiple for learning rate
            momentum_gamma - multiple for the momentum term
            regularization - multiple for L2 regularization
            gradReqd - boolean for updating weights. True == update weights

        returns:
            the trained model
        """
        if self.activation.activation_type != 'output':
            deltaCur = deltaCur[:, 1:]
        delta = self.activation.backward(self.a) * deltaCur
        # delta = np.mean(self.activation.backward(self.a) * deltaCur, axis=0)
        L2_penalty = 2 * regularization * self.w
        delta_next = delta @ self.w.T
        grad = -self.x.T @ delta + self.x.shape[0] * L2_penalty
        # grad = - np.outer(np.mean(self.x, axis=0), delta) + L2_penalty

        self.dw = - (learning_rate / self.x.shape[0]) * grad + momentum_gamma * self.dw
        
        if gradReqd:
            self.w = self.w + self.dw

        return delta_next
        
    def printLayer(self):
        print("Activation:", self.activation.activation_type)
        print("Weights:", np.shape(self.w)," weights \n")
        print(self.w)
        print("Output: \n")
        print(self.z)
        plt.imshow(self.w.reshape(np.shape(self.w)[1],np.shape(self.w)[0]))
        plt.show()
        plt.close()
        plt.clf()



class Neuralnetwork:
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.

    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.activations = config['activation']  # loads the layer list
        self.activations.append('output') # appends output activation for final layer
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None  # For saving the output vector of the model
        self.targets = None  # For saving the targets

        self.learning_rate = config['learning_rate'] # loads learning rate
        self.momentum_gamma = config['momentum_gamma'] # loads momentum factor
        self.regularization = config['L2_penalty'] # loads L2 penalty factor

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            self.layers.append(
                Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation(self.activations[i]),
                      config["weight_type"]))
            """if i < self.num_layers - 1:
                
            elif i == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output"),
                                         config["weight_type"]))"""

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        self.x = x
        for layer in self.layers:
            x = layer.forward(x)
        self.y = x

        if targets is not None:
            self.targets = targets
            return util.calculate_correct(self.y, targets), self.loss(self.y, targets)

    def loss(self, logits, targets):
        '''
        Calculates Binary Cross Entropy
        '''
        return - np.dot(targets.flatten(), np.log(logits).flatten()) / logits.shape[0]

    def backward(self, gradReqd=True):
        '''
        Call backward methods of individual layers.
        '''
        delta = self.targets - self.y
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate, self.momentum_gamma, self.regularization)

    def printLayerStructure(self):
        ct=0
        for layer in self.layers:
            print("Layer",ct, "weights:",np.shape(layer.w))
            print("     bias weight:",layer.w[0][0])
            print("     random weight",layer.w[4][8])
            ct+=1

    def forwardEpsilon(self,x,eps,layer_idx,biasORhidden,idx2,targets=None):
        self.x = x
        #set the specific layer weight to the w+e value
        self.layers[layer_idx].w[biasORhidden][idx2]=self.layers[layer_idx].w[biasORhidden][idx2]+eps
        for layer in self.layers:
            x = layer.forward(x)
        WplusE=x

        x = self.x
        #set the specific layer weight to the w-e value
        self.layers[layer_idx].w[biasORhidden][idx2]=self.layers[layer_idx].w[biasORhidden][idx2]-2*eps
        for layer in self.layers:
            x = layer.forward(x)
        WminE=x

        #reset the specific layer weight to its initial value
        self.layers[layer_idx].w[biasORhidden][idx2]=self.layers[layer_idx].w[biasORhidden][idx2]+eps

        if targets is not None:
            self.targets=targets
            return util.calculateCorrect(WplusE,targets), self.loss(WplusE,targets),util.calculateCorrect(WminE,targets), self.loss(WminE,targets)

