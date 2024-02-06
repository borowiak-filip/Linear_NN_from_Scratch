import numpy as np

def cv(ml_model, X, y, epochs, cv, lr):

  accuracy_test = []
  accuracy_train = []


  fold_size = X.shape[0] // cv
  X_fold = np.array_split(X, fold_size)
  y_fold = np.array_split(y, fold_size)


  for i in range(cv):

    correct_vals = 0
    all_vals = 0
    X_KFold_train = np.concatenate([X_fold[j] for j in range(cv+1) if j != i])
    y_KFold_train = np.concatenate([y_fold[j] for j in range(cv+1) if j != i])
    X_KFold_val = X_fold[i]
    y_KFold_val = y_fold[i]

    results = ml_model.train(X_KFold_train, y_KFold_train, epochs=epochs, lr_rate=lr, epoch_print=False)

    accuracy_train.append(results['Epochs_accuracy'][0])

    for k in range(cv):

      y_pred = ml_model.predict(X_KFold_val[k])

      if np.argmax(y_pred, axis=1) == y_KFold_val[k]:
        correct_vals += 1
      all_vals += 1

    accuracy_test.append(correct_vals/all_vals)
    print(f"Accuracy in fold: {i} = {(accuracy_test[i - 1] * 100)}%")

  print(f"Average Training Accuracy = {(np.mean(accuracy_train) * 100)}%")
  print(f"Average Testing Accuracy = {(np.mean(accuracy_test) * 100)}%")

  return (np.mean(accuracy_train) * 100), (np.mean(accuracy_test) * 100)