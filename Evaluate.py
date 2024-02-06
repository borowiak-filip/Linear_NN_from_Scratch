
import numpy as np

def eval(model, X_train, y_train, X_test, y_test, X_val, y_val):
    positive_predictions_train, negative_predictions_train, total_predictions_train  = model.EvaluateScore(X_train, y_train)

    positive_predictions_test, negative_predictions_test, total_predictions_test  = model.EvaluateScore(X_test, y_test)

    positive_predictions_val, negative_predictions_val, total_predictions_val  = model.EvaluateScore(X_val, y_val)

    print(" -- Train Set Score -- ")
    print(f"Good Predictions: {positive_predictions_train}")
    print(f"Wrong Predictions: {total_predictions_train - positive_predictions_train}")
    print(f"All Predictions: {total_predictions_train}")
    print(f"Accuracy: {np.round((positive_predictions_train/total_predictions_train) * 100)} % \n")

    print(" -- Test Set Score -- ")
    print(f"Good Predictions: {positive_predictions_test}")
    print(f"Wrong Predictions: {total_predictions_test - positive_predictions_test}")
    print(f"All Predictions: {total_predictions_test}")
    print(f"Accuracy: {np.round((positive_predictions_test/total_predictions_test) * 100)} % \n")

    print(" -- Validation Set Score -- ")
    print(f"Good Predictions: {positive_predictions_val}")
    print(f"Wrong Predictions: {total_predictions_val - positive_predictions_val}")
    print(f"All Predictions: {total_predictions_val}")
    print(f"Accuracy: {np.round((positive_predictions_val/total_predictions_val) * 100)} % \n")