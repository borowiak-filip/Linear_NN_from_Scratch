from Engine.Interfaces.ILayer import ILayer
from Engine.Activations import Softmax, Sigmoid
import numpy as np

# Dense Layer Class
class DenseLayer(ILayer):

    def __init__(self, n_in, n_out, activation, index):
        self.n_in = n_in # number of inputs to the layer
        self.n_out = n_out #number of "Neurons" in the layer -> input to the next layer
        self.weights = np.random.randn(n_in, n_out)
        self.bias = np.zeros((1, n_out))
        self.name = "Dense Layer"
        self.dim = (n_in, n_out) # Dimensions of the layer
        self.activation = activation
        self.Y_true = None # True value for the input
        self.x_in = None # Input to the layer
        self.out = None # Output from the layer
        self.index = index
        self.dW = None # derivative of weights
        self.db = None #derivative of bias
        np.random.seed(42)

        if activation == "Softmax":
            self.nonLinearFunction = Softmax()
        elif activation == "Sigmoid":
            self.nonLinearFunction = Sigmoid()

    def resetLayer(self):

        self.weights = np.random.randn(self.n_in, self.n_out)
        self.bias = np.zeros((1, self.n_out))

    def params(self):

        return np.size(self.weights) + np.sum(self.bias)

    def summary(self):

        return self.name, self.dim, self.activation

    def __call__(self, x_in):
        self.x_in = x_in
        xw = np.dot(x_in, self.weights) + self.bias
        g = self.nonLinearFunction.applyActivation(xw)
        self.out = g
        return g

    def predict(self, x_in):
        if self.index == 1:
            self.x_in = np.array(x_in)
        else:
            self.x_in = x_in
        xw = np.dot(x_in, self.weights) + self.bias
        g = self.nonLinearFunction.applyActivation(xw)
        self.out = g
        return g

    def update(self, lr):
        self.weights -= lr * self.dW
        self.bias -= lr * self.db

    def backward(self, out_grad, weights_k, m):

      # Backpropgation pass for the layer
      # out_grad is the derivative dZ from the previous layer

        if self.activation == "Softmax":

            self.dW = 1 / m * np.dot(self.x_in.T, out_grad)
            self.db = 1 / m * np.sum(out_grad, axis=0, keepdims=True)

        else:
            #dZ = Wk+1.T * outgrad * dG
            dZ = np.dot(out_grad, weights_k.T)
            dG = self.nonLinearFunction.backward(np.dot(self.x_in, self.weights))
            out_grad = dZ * dG

            #if first layer
            if self.index == 1:
                self.dW = 1/m * np.dot(np.expand_dims(self.x_in, axis=1), out_grad)
                self.db = 1 / m * np.sum(out_grad, axis=0, keepdims=True)

            else:
                self.dW = 1 / m* np.dot(self.x_in.T, out_grad)
                self.db = 1 / m * np.sum(out_grad, axis=0, keepdims=True)


        return out_grad, self.weights





