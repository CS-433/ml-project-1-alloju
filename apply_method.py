from sys import implementation
from utilities import compute_mse,compute_rmse, sigmoid
from paths import  prediction_dir
import os.path as op
import numpy as np
from helpers import create_csv_submission

def apply_method(method,y_tr,x_tr,y_val,x_val, x_te, id):
    #TODO: def applay_method(method, y_tr, x_tr, y_val, x_val, x_te, id, lamda_ = 0.5, initial_w = np.array([0]), max_iters = 100, gamma = 0.1):

    """Apply a given method to the training and validation sets.

    Args:
        method: method to apply to the data
        y_tr: training labels
        x_tr : training features
        y_val : validation labels
        x_val : validation features

    Returns:
        rmse_tr: training rmse
        rmse_val: validation rmse
    """

    # TODO: if blablabla in file name 
    #une manière plus élégante de faire maybe ?:
    #import foo
    #bar = getattr(foo, 'bar')
    #result = bar()

    if ('mean_squared_error' in str(method) or 'mean_squared_error_sgd' in str(method) or 'logistic_regression' in str(method)):
        weights = np.random.rand(x_tr.shape[1])
        w, mse = method(y_tr,x_tr, initial_w = weights, max_iters = 100, gamma = 0.1)
    elif ('least_squares' in str(method)):
        w, mse = method(y_tr, x_tr)
    elif ('ridge_regression' in str(method)):
        w, mse = method(y_tr, x_tr, lambda_ = 0.1)
    elif ('reg_logistic_regression' in str(method)):
        weights = np.random.rand(x_tr.shape[1])
        w, mse = method(y_tr, x_tr, lambda_ = 0.1, initial_w = weights, max_iters = 100, gamma = 0.1)
    rmse_tr = compute_rmse(mse)
    rmse_val = compute_rmse(compute_mse(y_val,x_val,w))
    predict(method, id, x_te, w)
    return rmse_tr, rmse_val

def predict(method, id, x_te, w):
    """_summary_

    Args:
        method (_type_): _description_
        id (_type_): _description_
        x_te (_type_): _description_
    """
    y = np.dot(x_te,w)
    # appliquer les labels
    y = sigmoid(y)
    y[y < 0.5] = -1
    y[y > 0.5] = 1
    #pred = np.vstack((np.array(["Id", "Prediction"]),np.column_stack((id,y))))
    path = op.join(prediction_dir, "prediction" + str(method) + ".csv")
    create_csv_submission(id, y, path)
    #np.savetxt(path, pred, delimiter=",")