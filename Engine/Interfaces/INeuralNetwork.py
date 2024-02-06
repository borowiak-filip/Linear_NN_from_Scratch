import abc

# Neural Netowork interface
class INeuralNetwork(abc.ABC):

    def __init__(self, layers):
        self.layers = layers

    @abc.abstractmethod
    def getParams(self):
      # Returns number of parameters in the model
        pass

    @abc.abstractmethod
    def networkSummary(self):
      # Returns architecture description and number of parameters per layer
        pass

    @abc.abstractmethod
    def predict(self, x_in):
      # Runs single prediction
        pass

    @abc.abstractmethod
    def train(self, X, Y, lr_rate, epochs, plots=False, epoch_print=True):
      # Trains the model
        pass
    @abc.abstractmethod
    def GetNetworkWeights(self):
      # Returns list of weigths and biases for each layer
        pass

    @abc.abstractmethod
    def ResetNetwork(self):
      # Resets weights and biases
        pass

    @abc.abstractmethod
    def EvaluateScore(self, X, Y):
      # Runs prediction score
      pass

    @abc.abstractmethod
    def ModelTrainingPerformanceMetrics(self):
      # Plots model training performance
      pass

    @abc.abstractmethod
    def plotGradientDescent(self):
      # Plots Gradient Descent of the training
      pass
