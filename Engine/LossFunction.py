import numpy as np

# Catagorical Cross Entropy Loss Function
class CEL():
    def __init__(self):
        self.name = "Cross Entropy Loss"
        def CrossEntropyLoss(y_ground_truth, logits):
            logits = np.clip(logits, 1e-7, 1 - 1e-7) # Clipping values; prevents from large values
            m = logits.shape[0]

            if len(y_ground_truth.shape) == 1: # for vectors
                confidence = logits[range(m), y_ground_truth]

            elif len(y_ground_truth.shape) == 2: # for one hot encodings
                confidence = np.sum(logits*y_ground_truth, axis=1)

            return - np.log(confidence) # Returns negaitve log of the confidence of the logits

        def Derivative_CrossEntropyLoss(y_ground_truth, logits):
            return logits - y_ground_truth # Derivative of the Cross Entropy Loss == Derivative of Softmax

        self._lossFunction = CrossEntropyLoss
        self._lossFunctionDerivative = Derivative_CrossEntropyLoss