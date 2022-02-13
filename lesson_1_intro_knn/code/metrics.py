import numpy as np
import pandas as pd


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    tp = sum(np.logical_and((y_pred == y_true), (y_true == '1')))
    tn = sum(np.logical_and((y_pred == y_true), (y_true == '0')))
    fp = sum(np.logical_and((y_pred != y_true), (y_true == '0')))
    fn = sum(np.logical_and((y_pred != y_true), (y_true == '1')))
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1, accuracy
    
    
    """
    YOUR CODE IS HERE
    """
    pass


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    acc = (y_pred == y_true)
    accuracy = sum(acc)/len(y_true)
    return accuracy
    """
    YOUR CODE IS HERE
    """
    pass


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """
    y_pred = pd.Series(data=y_pred, index=y_true.index.values.tolist())
    r2 = 1 - sum((y_true[i] - y_pred[i])**2 for i in y_true.index.values.tolist()) / sum((y_true[i] - y_true.mean())**2 for i in y_true.index.values.tolist())
    return r2
    """
    YOUR CODE IS HERE
    """
    pass


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """
    y_pred = pd.Series(data=y_pred, index=y_true.index.values.tolist())
    mse = 1 / len(y_true) * sum((y_true[i] - y_pred[i])**2 for i in y_true.index.values.tolist())
    return mse
    """
    YOUR CODE IS HERE
    """
    pass


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """
    y_pred = pd.Series(data=y_pred, index=y_true.index.values.tolist())
    mae = 1 / len(y_true) * sum(abs(y_true[i] - y_pred[i]) for i in y_true.index.values.tolist())
    return mae
    """
    YOUR CODE IS HERE
    """
    pass
    