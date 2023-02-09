import numpy as np
import matplotlib.pyplot as plt
import util

class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO in case you want to add variables here
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU","output"]:   #output can be used for the final layer. Feel free to use/remove it
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
        Implement the sigmoid activation here.
        """
        return 1/(1+np.exp(-x))
        #raise NotImplementedError("Sigmoid not implemented")

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return np.tanh(x)
        #raise NotImplementedError("Tanh not implemented")

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        return np.array([(x[i]>0)*x[i] for i in range(len(x))])
        #raise NotImplementedError("ReLU not implemented")

    def output(self, x):
        """
        Implement softmax function here.
        Remember to take care of the overflow condition.
        TOO BIG
        """
        denom = np.sum([np.e**(aa) for aa in x])
        return np.exp(x)/denom
        #raise NotImplementedError("output activation not implemented")

    def grad_sigmoid(self,x):
        """
        Compute the gradient for sigmoid here.
        """
        return np.exp(-x)/(1+np.exp(-x))**2
        #raise NotImplementedError("Sigmoid gradient not implemented")

    def grad_tanh(self,x):
        """
        Compute the gradient for tanh here.
        """
        return 1-self.tanh(x)**2
        #raise NotImplementedError("Tanh gradient not implemented")

    def grad_ReLU(self,x):
        """
        Compute the gradient for ReLU here.
        """
        return np.array([(x[i]>0) for i in range(len(x))])
        #raise NotImplementedError("ReLU gradient not implemented")

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """
        return 1


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation, weightType):
        """
        TODO in case you want to add variables here
        Define the architecture and create placeholders.
        """
        np.random.seed(42)

        self.w = None
        if (weightType == 'random'):
            self.w = 0.01 * np.random.random((in_units + 1, out_units))

        self.x = None    # Save the input to forward in this
        self.a = None    #output without activation
        self.z = None    # Output After Activation
        self.activation = activation   #Activation function


        self.dw = 0  # Save the gradient w.r.t w in this. You can have bias in w itself or uncomment the next line and handle it separately
        # self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
         Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """
        self.x = util.append_bias(x)
        self.a = self.x @ self.w
        self.z = self.activation.forward(self.a)
        return self.z
        

    def backward(self, deltaCurrent, learning_rate, momentum_gamma, regularization, gradReqd=True):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input and
        computes gradient for its weights and the delta to pass to its previous layers. gradReqd is used to specify whether to update the weights i.e. whether self.w should
        be updated after calculating self.dw
        The delta expression (that you prove in PA2 part1) for any layer consists of delta and weights from the next layer and derivative of the activation function
        of weighted inputs i.e. g'(a) of that layer. Hence deltaCur (the input parameter) will have to be multiplied with the derivative of the activation function of the weighted
        input of the current layer to actually get the delta for the current layer. Remember, this is just one way of interpreting it and you are free to interpret it any other way.
        Feel free to change the function signature if you think of an alternative way to implement the delta calculation or the backward pass.
        gradReqd=True means update self.w with self.dw. gradReqd=False can be helpful for Q-3b
        """
        if self.activation.activation_type != 'output':
            deltaCurrent=deltaCurrent[:,1:]
        
        g_deriv = self.activation.backward(self.a)
        delta = g_deriv*deltaCurrent
        deltaNext=delta @ self.w.T

        L2_penalty = 2 * regularization * self.w
        grad = -self.x.T @ delta + L2_penalty*len(self.x)
        self.dw = - (learning_rate/len(self.x)) * grad + momentum_gamma * self.dw

        if gradReqd==False: # for 3b
            self.dw = -1*(learning_rate/len(self.x))*grad

        if gradReqd:
            self.w=self.w+self.dw

        return deltaNext



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

class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.

    """

    def __init__(self, config):
        """
        TODO in case you want to add variables here
        Create the Neural Network using config. Feel free to add variables here as per need basis
        """
        self.layers = []  # Store all layers in this list.
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None        # For saving the output vector of the model
        self.targets = None  # For saving the targets

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(
                    Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation(config['activation']),
                          config["weight_type"]))
            elif i == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output"),
                                         config["weight_type"]))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)


    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        self.x = x
        for layer in self.layers:
            x = layer.forward(x)
        self.y = x
        
        if targets is not None:
            self.targets=targets
            return util.calculateCorrect(self.y,targets), self.loss(self.y,targets)
        
        #raise NotImplementedError("Forward propagation not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        return -np.sum(np.log(logits)*targets)/len(logits)
        

        #raise NotImplementedError("Loss not implemented for NeuralNetwork")

    def backward(self, gradReqd=True):
        '''
        TODO: Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        '''
        delta=self.targets-self.y
        for layer in reversed(self.layers):
            delta = layer.backward(delta,self.learning_rate, self.mom_gamma, self.L2penalty, gradReqd)
        #raise NotImplementedError("Backward propagation not implemented for NeuralNetwork")

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



