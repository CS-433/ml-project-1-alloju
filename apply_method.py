from sys import implementation
from utilities import compute_accuracy, compute_mse,compute_rmse, sigmoid
from paths import  prediction_dir
import os.path as op
import numpy as np
from helpers import create_csv_submission

def apply_method(method,y_tr,x_tr,y_val = np.zeros([10,1]) ,x_val = np.zeros([10,1]), x_te = np.zeros([5,1]), id = np.zeros(5), lambda_ = 0.5, initial_w = None, max_iters = 100, gamma = 0.1, cross_val = False, validation = True, separation = False):

    """Apply a given method to the training and validation sets.

    Args:
        method: method to apply to the data
        y_tr: training labels
        x_tr: training features
        y_val: validation labels
        x_val: validation features
        x_te:  

    Returns:
        rmse_tr: training rmse
        rmse_val: validation rmse
        w: computed weights
    """
    print('xtr.shape', x_tr.shape)
    # TODO: if blablabla in file name 
    #une manière plus élégante de faire maybe ?:
    #import foo
    #bar = getattr(foo, 'bar')
    #result = bar()

    if (initial_w == None):
        initial_w = np.zeros(x_tr.shape[1])

    if ('reg_logistic_regression' in str(method)):
        w, neg_log_likelihood = method(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
        mse = compute_mse(y_tr, x_tr, w)
    elif ('logistic_regression' in str(method)):
        w, neg_log_likelihood = method(y_tr,x_tr, initial_w, max_iters, gamma)
        mse = compute_mse(y_tr, x_tr, w)
    elif ('mean_squared_error' in str(method) or 'mean_squared_error_sgd' in str(method)):
        w, mse = method(y_tr,x_tr, initial_w, max_iters, gamma)
    elif ('least_squares' in str(method)):
        w, mse = method(y_tr, x_tr)
    elif ('ridge_regression' in str(method)):
        w, mse = method(y_tr, x_tr, lambda_)
    
    mse_tr = mse
    mse_val = 0 #avoid error if no validation set
    acc_val = 0
    if validation:
        mse_val = compute_mse(y_val,x_val,w)
        acc_val = compute_accuracy(y_val,x_val,w)

    if not(cross_val):
        if(separation):
            y_bin = predict(method, id, x_te, w, separation)
        else: 
            predict(method, id, x_te, w)
    acc_train = compute_accuracy(y_tr, x_tr, w)

    if(separation):
        return mse_tr, mse_val, y_bin
    else:
        return mse_tr, mse_val #return acc_train, acc_val
    

def predict(method, id, x_te, w, separation = False):
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

    if(separation):
        return y_bin
    else:
        #pred = np.vstack((np.array(["Id", "Prediction"]),np.column_stack((id,y))))
        path = op.join(prediction_dir, "prediction" + str(method) + ".csv")
        create_csv_submission(id, y_bin, path)
        #np.savetxt(path, pred, delimiter=",")

def joining_prediction(method, id, y):
    path = op.join(prediction_dir, "prediction" + str(method) + ".csv")
    create_csv_submission(id, y, path)

        