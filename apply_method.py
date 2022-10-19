from sys import implementation
from utilities import compute_mse,compute_rmse, sigmoid
from paths import  prediction_dir
import os.path as op
import numpy as np
from helpers import create_csv_submission

def apply_method(method,y_tr,x_tr,y_val,x_val, x_te, id, lambda_ = 0.5, initial_w = None, max_iters = 100, gamma = 0.1):

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

    if (initial_w == None):
        initial_w = np.zeros(x_tr.shape[0])

    if ('mean_squared_error' in str(method) or 'mean_squared_error_sgd' in str(method) or 'logistic_regression' in str(method)):
        w, mse = method(y_tr,x_tr, initial_w, max_iters, gamma)
    elif ('least_squares' in str(method)):
        w, mse = method(y_tr, x_tr)
    elif ('ridge_regression' in str(method)):
        w, mse = method(y_tr, x_tr, lambda_)
    elif ('reg_logistic_regression' in str(method)):
        w, mse = method(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
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
    y_bin = sigmoid(y)
    y_bin[y_bin < 0.5] = -1
    y_bin[y_bin >= 0.5] = 1
    #pred = np.vstack((np.array(["Id", "Prediction"]),np.column_stack((id,y))))
    path = op.join(prediction_dir, "prediction" + str(method) + ".csv")
    create_csv_submission(id, y_bin, path)
    #np.savetxt(path, pred, delimiter=",")