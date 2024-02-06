import numpy as np

def shuffle(df):
    # Shuffle the data
    for k in range(5):
        df= df.sample(frac=1).reset_index(drop=True)
    return df

# standard Scaling
def standardScaler(X):
  u = np.mean(X, axis=0)
  s = np.std(X, axis=0)
  X
  for idx, row in enumerate(X):
    X[idx] = (X[idx] - u) / s

  return X


def split(X, y):

    train_size = int(0.7*X.shape[0])
    test_size = int(0.9*X.shape[0])

    X_train = X[:train_size, :,]
    y_train = y[:train_size]
    X_test = X[train_size:test_size, :,]
    y_test = y[train_size:test_size]
    X_val = X[test_size:, :,]
    y_val = y[test_size:]

    return X_train, y_train, X_test, y_test, X_val, y_val
