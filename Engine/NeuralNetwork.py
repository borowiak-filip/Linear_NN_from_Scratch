from Engine.Interfaces.INeuralNetwork import INeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

class NeuralNetwork(INeuralNetwork):
    def __init__(self, layers, loss, batch_size=None, lr_decay=False, decay_drop=100, decay_factor=0.5):
        self.layers = layers # layers in the model
        self.batch_size = batch_size
        self.loss = loss # loss function of the model
        self.decay = lr_decay # learning rate decay on / off
        self.decay_drop = decay_drop # decay drop step
        self.decay_factor = decay_factor # decay division factor
        self.lr = None # learning rate of the model

        # Evaluation attributes
        self.losses_count = []
        self.epochs_count = []
        self.train_epochs_accuracy = []
        # Setting seed to 42 for reproducibility
        np.random.seed(42)


    def getParams(self):
        total = 0
        print("-------------------------------------------------------------")
        for layer in self.layers:
            val = layer.params()
            if isinstance(val, float):
                total += val
        print(f"Total number of paramteters = {int(total)}")
        print("-------------------------------------------------------------")


    def networkSummary(self):
        print(" -- Model Loss function -- \n")

        print(f"|- {self.loss.name} \n")
        print("-- Model Architecture -- \n")
        print("-------------------------------------------------------------")
        for layer in self.layers:
            name, dim, activation = layer.summary()
            if activation is not None:
                print(f"Layer: {name}, Shape: {dim}, Neurons: {dim[1]}")
                print("|")
                print(f"|- Activation Function: {activation}")
                print("-------------------------------------------------------------")
            else:
                print(f"Layer: {name}, Shape: {dim}, Neurons: {dim[1]}")
                print("-------------------------------------------------------------")

    def ModelTrainingPerformanceMetrics(self):
        if self.decay:
            lr = f"lr: {self.lr}, decay_factor: {self.decay_factor}"
        else:
            lr = self.lr

        def lossPerEpoch():
            plt.plot(self.epochs_count, self.losses_count)
            plt.ylabel('Loss function')
            plt.xlabel('Epochs')
            plt.grid(visible=True)
            plt.title(f"Loss function per Epoch | Lr = {lr} ")
            best_loss = min(self.losses_count)
            plt.axhline(y=best_loss, color='black', linestyle='--', label=f'Best Loss = {best_loss}')
            plt.show()

        def PlotAccuracyPerEpoch():
            plt.plot(self.epochs_count, self.train_epochs_accuracy)
            plt.ylabel('Accuracy')
            plt.xlabel('Epochs')
            plt.grid(visible=True)
            plt.title(f"Accuracy per Epoch | Lr = {lr}")
            best_acc = max(self.train_epochs_accuracy)
            plt.axhline(y=best_acc, color='black', linestyle='--', label=f'Best Accuracy = {best_acc}')
            plt.show()

        lossPerEpoch()
        PlotAccuracyPerEpoch()

    def ResetNetwork(self):
        for layer in self.layers:
            layer.resetLayer()

    def GetNetworkWeights(self):
        model_weights = {}
        for layer in self.layers:
            model_weights[layer.index]={
                "Layer_Index": layer.index,
                "Layer_Name": layer.name,
                "Layer_Weights": layer.weights,
                "Layer_Bias": layer.bias}

        return model_weights


    def predict(self, x_in):
        out = x_in
        for layer in self.layers:
            out = layer.predict(out)
        return out

    def one_hot(self, Y):
      #one hot encode values 0, 1, 2 to be compatible with cross Entropy Loss
        if Y == 0:
            out = np.array([1, 0, 0])
        elif Y == 1:
            out = np.array([0, 1, 0])
        elif Y == 2:
            out = np.array([0, 0, 1])

        return out.reshape(-1, 1).T


    def train(self, X, Y, lr_rate, epochs, plots=False, epoch_print=True):
        self.losses_count = []
        self.epochs_count = []
        self.train_epochs_accuracy = []
        self.lr = lr_rate
        for epoch in range(epochs):
            if self.decay:
                if epoch > 0 and epoch % 100 == self.decay_drop:
                    self.lr *= self.decay_factor

            if self.batch_size is None or 1:
              # if not mini batches
                self.batch_size = 1
                Xb, Yb = X, Y
            else:
              # else create random permuation of batch size across rows
                ri = np.random.permutation(X.shape[0])[:self.batch_size]
                Xb, Yb = X[ri], Y[ri]


            total_predictions = len(Yb)
            positive_predictions = 0
            for idx, (x, y) in enumerate(zip(Xb, Yb)):

                # forward pass
                out = self.predict(Xb[idx])
                y = self.one_hot(Yb[idx])

                if np.argmax(out) == Yb[idx]:
                    positive_predictions += 1

                # Calculate loss
                loss = self.loss._lossFunction(y, out)

                #Backward Pass
                gradient = self.loss._lossFunctionDerivative(y, out)


                weights_k = None # Weights from the layer k+1 for backward pass
                for layer in reversed(self.layers):
                    gradient, weights_k = layer.backward(out_grad=gradient, weights_k=weights_k, m=self.batch_size)

                # Update gradient per layer
                for layer in self.layers:
                    layer.update(self.lr)

            '''
            if loss <= 0.000001:
                print(f"Loss function achieved: {loss[0]}")

                if plots:
                    self.ModelTrainingPerformanceMetrics()

                return ({
                "Losses": self.losses_count,
                "Epochs_accuracy": self.train_epochs_accuracy,
                  "Epochs_count": self.epochs_count
                })
                '''


            if epoch % 1 == 0:
                self.losses_count.append(loss[0])
                self.epochs_count.append(epoch)
                self.train_epochs_accuracy.append(positive_predictions/total_predictions)
                if epoch_print:
                    print(f"Epoch: {epoch}, loss: {loss[0]}")

        if plots:
            self.ModelTrainingPerformanceMetrics()

        return ({
                "Losses": self.losses_count,
                "Epochs_accuracy": self.train_epochs_accuracy,
                  "Epochs_count": self.epochs_count
                })

    def plotGradientDescent(self):
      # 3D print gradient Descent

      fig = plt.figure()
      ax = plt.figure().add_subplot(projection='3d')
      Xs = np.linspace(-8, 8, len(self.epochs_count))
      Ys = np.linspace(-4, 4, len(self.epochs_count))

      Xs, Ys = np.meshgrid(Xs, Ys)
      Z = np.array(self.losses_count).reshape(1, -1)

      surf = ax.plot_surface(Xs, Ys, Z, cmap=cm.coolwarm, rstride=2, cstride=2)
      plt.show()

    def EvaluateScore(self, X, Y):
        positive_predictions = 0
        negative_predictions = 0
        total_predictions = 0

        for idx, (x, y) in enumerate(zip(X, Y)):

            result =  self.predict(X[idx])

            if np.argmax(result) == Y[idx]:
               positive_predictions += 1

            else:
                negative_predictions += 1

            total_predictions += 1

        score = positive_predictions / total_predictions

        #print(f"Accuracy Score: {np.round(score*100)} %")

        return positive_predictions, negative_predictions, total_predictions








