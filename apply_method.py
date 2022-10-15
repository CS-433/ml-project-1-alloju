from utilities import compute_mse,compute_rmse
from paths import  prediction_dir
import os.path as op
import numpy as np

def apply_method(method,y_tr,x_tr,y_val,x_val, x_te, id):
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
    if (method == 'mean_squared_error' or 'mean_squared_error_sgd' or 'logistic_regression'):
        weights = np.random.rand(x_tr.shape[1])
        w, mse = method(y_tr,x_tr, initial_w = weights, max_iters = 100, gamma = 0.1)
    elif (method == 'least_squares'):
        w, mse = method(y_tr, x_tr)
    elif (method == 'ridge_regression'):
        w, mse = method(y_tr, x_tr, lambda_ = 0.1)
    elif (method == 'reg_logistic_regression'):
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
    print(x_te.shape)
    print(w.shape)
    y = np.dot(x_te,w)
    # appliquer les labels
    pred = np.column_stack((id,y))
    path = op.join(prediction_dir, "prediction" + str(method) + ".csv")
    np.savetxt(path, pred, delimiter=",")