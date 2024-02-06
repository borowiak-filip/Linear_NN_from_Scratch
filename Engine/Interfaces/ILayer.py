import abc
class ILayer(abc.ABC):
    def __init__(self):

        self.x_in = None
        self.weights = None
        self.bias = None
        self.name = None
        self.dim = None
        self.activation = None
        self.nonLinearFunction = None


    @abc.abstractmethod
    def params(self):
      # Returns number of parameters in the layer
        pass
    @abc.abstractmethod
    def summary(self):
      # Returns layer descripition and parameters
        pass

    @abc.abstractmethod
    def __call__(self, xIn):
        pass

    @abc.abstractmethod
    def predict(self, xIn):
      # Runs single prediction
        pass
    @abc.abstractmethod
    def backward(self, out_grad, lr):
      # Invokes backpropagation
        pass