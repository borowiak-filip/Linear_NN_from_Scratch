from Engine import DenseLayer as layer
from Engine import NeuralNetwork as nn
from Engine import LossFunction as lf
import PreProcessing as pp
from CrossValidation import cv
from Evaluate import eval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("-- data here --")

# Remove unnecessary columns
df.dropna(subset=['sex'], inplace=True)
df.drop(columns=df.columns[0], axis=1, inplace=True)
df.drop(columns=['sex', 'year', 'island'], axis=1, inplace=True)
columns = df.columns
# Now species data needs to be changed into numerical values
df['species']=df['species'].replace('Adelie', 0)
df['species']=df['species'].replace('Gentoo', 1)
df['species']=df['species'].replace('Chinstrap', 2)
targets_lookup = {0:'Adelie', 1:"Gentoo", 2:"Chinstrap"}

df = pp.shuffle(df)

# Seperate features from labels
X = df.drop(columns='species').to_numpy()
y = df['species'].to_numpy()

assert X.shape == (333, 4)
assert y.shape == (333, )

X = pp.standardScaler(X) # This is technically wrong, as you should scaled the data after the split to prevent from data leakage.

X_train, y_train, X_test, y_test, X_val, y_val = pp.split(X, y)


SLP_layout = [
    layer.DenseLayer(4, 16, "Sigmoid", index=1),
    layer.DenseLayer(16, 3, "Softmax", index=2)]

MLP_3_layout = [
    layer.DenseLayer(4, 16, "Sigmoid", index=1),
    layer.DenseLayer(16, 16, "Sigmoid", index=2),
    layer.DenseLayer(16, 3, "Softmax", index=3)]


MLP_8_layout = [
    layer.DenseLayer(4, 16, "Sigmoid", index=1),
    layer.DenseLayer(16, 16, "Sigmoid", index=2),
    layer.DenseLayer(16, 32, "Sigmoid", index=3),
    layer.DenseLayer(32, 32, "Sigmoid", index=4),
    layer.DenseLayer(32, 32, "Sigmoid", index=5),
    layer.DenseLayer(32, 16, "Sigmoid", index=6),
    layer.DenseLayer(16, 16, "Sigmoid", index=7),
    layer.DenseLayer(16, 3, "Softmax", index=8),
]


network_layout = [
    layer.DenseLayer(4, 16, "Sigmoid", index=1),
    layer.DenseLayer(16, 32, "Sigmoid", index=2),
    layer.DenseLayer(32, 64, "Sigmoid", index=3),
    layer.DenseLayer(64, 128, "Sigmoid", index=4),
    layer.DenseLayer(128, 256, "Sigmoid", index=5),
    layer.DenseLayer(256, 512, "Sigmoid", index=6),
    layer.DenseLayer(512, 512, "Sigmoid", index=7),
    layer.DenseLayer(512, 256, "Sigmoid", index=8),
    layer.DenseLayer(256, 256, "Sigmoid", index=9),
    layer.DenseLayer(256, 128, "Sigmoid", index=10),
    layer.DenseLayer(128, 128, "Sigmoid", index=11),
    layer.DenseLayer(128, 64, "Sigmoid", index=12),
    layer.DenseLayer(64,  64, "Sigmoid", index=13),
    layer.DenseLayer(64, 32, "Sigmoid", index=14),
    layer.DenseLayer(32, 32, "Sigmoid", index=15),
    layer.DenseLayer(32, 16, "Sigmoid", index=16),
    layer.DenseLayer(16, 16, "Sigmoid", index=17),
    layer.DenseLayer(16, 3, "Softmax", index=18)
]

def runCrossValidation():
    # Applying cross validation across 3 different network architectures and different configurations
    # Batches, learning rates

    batch_sizes = [4, 16, 32, 64]
    layers = [SLP_layout, MLP_3_layout, MLP_8_layout]
    lrs = [0.1, 0.01, 0.001, 0.0001]
    epochs = 100
    best_params = {"SLP": [0,1,1], # accuracy, learning rate, batch
                "MLP_3": [0,1,1],
                "MLP_8": [0,1,1]}

    for batch_size in batch_sizes:
        for lr in lrs:
            list_of_models = {"SLP": [0, 0, 0], # idx = 0
                        "MLP_3": [0, 0, 0], # idx = 1
                        "MLP_8": [0, 0, 0]} # idx = 2
            for idx, layer in enumerate(layers):
                cv_model = nn.NeuralNetwork(layer, lf.CEL(), batch_size=batch_size)
                print(f"-- CV for Batch Size: {batch_size} network Architecture : {idx} Learning Rate: {lr} --")
                mean_test_acc, mean_train_acc = cv(cv_model, X, y, epochs=epochs, cv=10, lr=lr)

                if mean_test_acc > best_params['SLP'][0] and idx == 0:
                    best_params['SLP'][0] = mean_test_acc
                    best_params['SLP'][1] = lr
                    best_params['SLP'][2] = batch_size

                if mean_test_acc >= best_params['MLP_3'][0] and idx == 1:
                    best_params['MLP_3'][0] = mean_test_acc
                    best_params['MLP_3'][1] = lr
                    best_params['MLP_3'][2] = batch_size

                if mean_test_acc >= best_params['MLP_8'][0] and idx == 1:
                    best_params['MLP_8'][0] = mean_test_acc
                    best_params['MLP_8'][1] = lr
                    best_params['MLP_8'][2] = batch_size

                print("-- End of Cross Validation -- \n")
                print("----------------------------------")

    print("Architecture - Accuracy - Learning Rate - Batch Size")
    print(best_params)

def runGridSearch(metrics):
    if metrics == "loss_function":
        best_params = {"SLP": [1,1,1], # loss, learning rate, batch
                    "MLP_3": [1,1,1],
                    "MLP_8": [1,1,1]}
        batch_sizes = [16, 32, 64, 128]
        layers = [SLP_layout, MLP_3_layout, MLP_8_layout]
        lrs = [0.1, 0.01, 0.001, 0.0001]
        epochs = 100
        for batch_size in batch_sizes:
            for lr in lrs:
                list_of_models = {"SLP": [0, 0, 0],
                            "MLP_3": [0, 0, 0],
                            "MLP_8": [0, 0, 0]}
                for idx, layer in enumerate(layers):
                    model = nn.NeuralNetwork(layer, lf.CEL(), batch_size=batch_size)
                    model.ResetNetwork()
                    x = model.train(X_train, y_train, plots=False, epoch_print=False, lr_rate=lr, epochs=epochs)
                    if idx == 0:
                        list_of_models['SLP'][0] = (x['Losses'])
                        list_of_models['SLP'][1] = (x['Epochs_accuracy'])
                        list_of_models['SLP'][2] = (x['Epochs_count'])

                    elif idx == 1:
                        list_of_models['MLP_3'][0] = (x['Losses'])
                        list_of_models['MLP_3'][1] = (x['Epochs_accuracy'])
                        list_of_models['MLP_3'][2] = (x['Epochs_count'])

                    elif idx == 2:
                        list_of_models['MLP_8'][0] = (x['Losses'])
                        list_of_models['MLP_8'][1] = (x['Epochs_accuracy'])
                        list_of_models['MLP_8'][2] = (x['Epochs_count'])



                plt.plot(list_of_models['SLP'][2], list_of_models['SLP'][0], label='SLP', color='blue')
                best_loss_SLP = min(list_of_models['SLP'][0])
                plt.axhline(y=best_loss_SLP, color='blue', linestyle='--', label=f'Best loss = {best_loss_SLP}')
                if best_loss_SLP < best_params['SLP'][0]:
                    best_params['SLP'][0] = best_loss_SLP
                    best_params['SLP'][1] = lr
                    best_params['SLP'][2] = batch_size


                plt.plot(list_of_models['MLP_3'][2], list_of_models['MLP_3'][0], label='MLP_3', color='green')
                best_loss_MLP_3 = min(list_of_models['MLP_3'][0])
                plt.axhline(y=best_loss_MLP_3, color='green', linestyle='--', label=f'Best loss = {best_loss_MLP_3}')
                if best_loss_MLP_3 < best_params['MLP_3'][0]:
                    best_params['MLP_3'][0] = best_loss_MLP_3
                    best_params['MLP_3'][1] = lr
                    best_params['MLP_3'][2] = batch_size



                plt.plot(list_of_models['MLP_8'][2], list_of_models['MLP_8'][0], label='MLP_8', color='red')
                best_loss_MLP_8 = min(list_of_models['MLP_8'][0])
                plt.axhline(y=best_loss_MLP_8, color='red', linestyle='--', label=f'Best loss = {best_loss_MLP_8}')
                if best_loss_MLP_8 < best_params['MLP_8'][0]:
                    best_params['MLP_8'][0] = best_loss_MLP_8
                    best_params['MLP_8'][1] = lr
                    best_params['MLP_8'][2] = batch_size

                plt.legend()

                plt.xlabel('Epochs')
                plt.ylabel('Loss function')
                plt.title(f'Training Loss for SLP vs MLP_3 vs MLP_8 for lr: {lr} | Batch Size: {batch_size}')
                plt.grid(True)

                plt.show()

        print("Architecture - Loss - Learning Rate - Batch Size")
        print(best_params)


    elif metrics == "accuracy":

        # Analysis of the training Accuracy for different learning rates with different batches
        best_params = {"SLP": [0,1,1], # accuracy, learning rate, batch
                    "MLP_3": [0,1,1],
                    "MLP_8": [0,1,1]}

        batch_sizes = [16, 32, 64, 128]
        layers = [SLP_layout, MLP_3_layout, MLP_8_layout]
        lrs = [0.1, 0.01, 0.001, 0.0001]
        epochs = 100
        for batch_size in batch_sizes:
            for lr in lrs:
                list_of_models = {"SLP": [0, 0, 0],
                            "MLP_3": [0, 0, 0],
                            "MLP_8": [0, 0, 0]}
                for idx, layer in enumerate(layers):
                    model = nn.NeuralNetwork(layer, lf.CEL(), batch_size=batch_size)
                    model.ResetNetwork()
                    x = model.train(X_train, y_train, plots=False, epoch_print=False, lr_rate=lr, epochs=epochs)
                    if idx == 0:
                        list_of_models['SLP'][0] = (x['Losses'])
                        list_of_models['SLP'][1] = (x['Epochs_accuracy'])
                        list_of_models['SLP'][2] = (x['Epochs_count'])

                    elif idx == 1:
                        list_of_models['MLP_3'][0] = (x['Losses'])
                        list_of_models['MLP_3'][1] = (x['Epochs_accuracy'])
                        list_of_models['MLP_3'][2] = (x['Epochs_count'])

                    elif idx == 2:
                        list_of_models['MLP_8'][0] = (x['Losses'])
                        list_of_models['MLP_8'][1] = (x['Epochs_accuracy'])
                        list_of_models['MLP_8'][2] = (x['Epochs_count'])



                plt.plot(list_of_models['SLP'][2], list_of_models['SLP'][1], label='SLP', color='blue')
                best_acc_SLP = max(list_of_models['SLP'][1])
                plt.axhline(y=best_acc_SLP, color='blue', linestyle='--', label=f'Best Accuracy = {best_acc_SLP}')
                if best_acc_SLP > best_params['SLP'][0]:
                    best_params['SLP'][0] = best_acc_SLP
                    best_params['SLP'][1] = lr
                    best_params['SLP'][2] = batch_size


                plt.plot(list_of_models['MLP_3'][2], list_of_models['MLP_3'][1], label='MLP_3', color='green')
                best_acc_MLP_3 = max(list_of_models['MLP_3'][1])
                plt.axhline(y=best_acc_MLP_3, color='green', linestyle='--', label=f'Best Accuracy = {best_acc_MLP_3}')
                if best_acc_MLP_3 > best_params['MLP_3'][0]:
                    best_params['MLP_3'][0] = best_acc_MLP_3
                    best_params['MLP_3'][1] = lr
                    best_params['MLP_3'][2] = batch_size


                plt.plot(list_of_models['MLP_8'][2], list_of_models['MLP_8'][1], label='MLP_8', color='red')
                best_acc_MLP_8 = max(list_of_models['MLP_8'][1])
                plt.axhline(y=best_acc_MLP_8, color='red', linestyle='--', label=f'Best Accuracy = {best_acc_MLP_8}')
                if best_acc_MLP_8 > best_params['MLP_8'][0]:
                    best_params['MLP_8'][0] = best_acc_MLP_8
                    best_params['MLP_8'][1] = lr
                    best_params['MLP_8'][2] = batch_size

                plt.legend()

                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.title(f'Training Accuracy for SLP vs MLP_3 vs MLP_8 for lr: {lr}  | Batch Size: {batch_size}')
                plt.grid(True)

                plt.show()

        print("Architecture - Accuracy - Learning Rate - Batch Size")
        print(best_params)

    else:
        print("Unkown metrics")

model = nn.NeuralNetwork(network_layout, lf.CEL(), batch_size=None)
model.getParams()
model.networkSummary()

model.train(X_train, y_train,lr_rate=0.001, epochs=10, plots=True)

model.plotGradientDescent()
eval(model, X_train, y_train, X_test, y_test, X_val, y_val)
print("Done")



model2 = nn.NeuralNetwork(network_layout, lf.CEL(), batch_size=32, decay_drop=100, decay_factor=0.5, lr_decay=True)
model.getParams()
model.networkSummary()

model2.train(X_train, y_train, lr_rate=0.01, epochs=10, plots=True)
model2.plotGradientDescent()
eval(model2, X_train, y_train, X_test, y_test, X_val, y_val)
print("Done")


runCrossValidation()
runGridSearch("loss_function")
runGridSearch("accuracy")
