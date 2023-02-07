import numpy as np
import util


class Activation:
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type="sigmoid"):
        """
        TODO in case you want to add variables here
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
        """
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        """
        return np.maximum(0, x)

    def output(self, x):
        """
        Remember to take care of the overflow condition.
        """
        return np.exp(x - np.max(x, axis=1).reshape(-1, 1)) / np.sum(np.exp(x - np.max(x, axis=1).reshape(-1, 1)), axis=1)[:, None]

    def grad_sigmoid(self, x):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        return np.exp(-x) / ((1 + np.exp(-x)) ** 2)

    def grad_tanh(self, x):
        """
        TODO: Compute the gradient for tanh here.
        """
        return 1 - np.tanh(x) ** 2

    def grad_ReLU(self, x):
        """
        TODO: Compute the gradient for ReLU here.
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
        TODO in case you want to add variables here
        Define the architecture and create placeholders.
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
        TODO: Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """
        self.x = util.append_bias(x)
        self.a = self.x @ self.w
        self.z = self.activation.forward(self.a)
        return self.z

    def backward(self, deltaCur, learning_rate, momentum_gamma, regularization, gradReqd=True):
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



class Neuralnetwork:
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.

    """

    def __init__(self, config):
        """
        TODO in case you want to add variables here
        Create the Neural Network using config. Feel free to add variables here as per need basis
        """
        self.layers = []  # Store all layers in this list.
        self.activations = config['activation']
        self.activations.append('output')
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None  # For saving the output vector of the model
        self.targets = None  # For saving the targets

        self.learning_rate = config['learning_rate']
        self.momentum_gamma = config['momentum_gamma']
        self.regularization = config['L2_penalty']

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
        TODO: Compute forward pass through all the layers in the network and return the loss.
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
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        return - np.dot(targets.flatten(), np.log(logits).flatten()) / logits.shape[0]

    def backward(self, gradReqd=True):
        '''
        TODO: Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        '''
        delta = self.targets - self.y
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate, self.momentum_gamma, self.regularization)

