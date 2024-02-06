import abc

# Activation function interface
class IActivation(abc.ABC):
    def __init__(self):

        self.x_in = None
        self.out = None

    @abc.abstractmethod
    def applyActivation(self, x_in):
        pass
    @abc.abstractmethod
    def backward(self, out_grad):
        pass