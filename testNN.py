from Engine import DenseLayer as layer
from Engine import NeuralNetwork as nn
from Engine import LossFunction as lf
import numpy as np


layout = [
    layer.DenseLayer(4, 16, "Sigmoid", index=1),
    layer.DenseLayer(16, 16, "Sigmoid", index=2),
    layer.DenseLayer(16, 32, "Sigmoid", index=3),
    layer.DenseLayer(32, 32, "Sigmoid", index=4),
    layer.DenseLayer(32, 32, "Sigmoid", index=5),
    layer.DenseLayer(32, 16, "Sigmoid", index=6),
    layer.DenseLayer(16, 16, "Sigmoid", index=7),
    layer.DenseLayer(16, 3, "Softmax", index=8),
]

model = nn.NeuralNetwork(layout, lf.CEL(), batch_size=None)


model.networkSummary()

model.getParams()

print(model.GetNetworkWeights())

X_train_dummy = np.random.randn(4, 4)
y_train_dummy = [1, 0, 1, 2]


model = nn.NeuralNetwork(layout, lf.CEL(), batch_size=32, decay_drop=100, decay_factor=0.5, lr_decay=True)
model.train(X_train_dummy, y_train_dummy, lr_rate=0.01, epochs=100, plots=True)

model.plotGradientDescent()