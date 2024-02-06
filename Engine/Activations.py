from Engine.Interfaces.IActivation import IActivation
import numpy as np

# Softmax "layer" for classification prediction
class Softmax(IActivation):
    def __init__(self):
        self.x_in = None # input to the layer
        self.out = None # output from the layer
        self.name = "Softmax"

    def applyActivation(self, x_in):
        self.x_in = x_in
        exp_vals = np.exp((x_in) - np.max(x_in))
        logits = exp_vals / np.sum(exp_vals)
        self.out = logits

        return logits

    def backward(self, out_grad, y_true):
      # With Cross Entropy Loss derivative of the Softmax is covered by (A - Y)
        pass


class Sigmoid(IActivation):
    def __init__(self):
        self.x_in = None # input to the layer
        self.out = None #output from the layer
        self.name = "Sigmoid"

    def applyActivation(self, x_in):
        self.x_in = x_in
        self.out = 1 / (1 + np.exp(-x_in))
        return self.out

    def backward(self, x_in):
        # (1 - sigmoid) * sigmoid -> derivative
        g = (1 - self.applyActivation(x_in)) * self.applyActivation(x_in)
        return g

