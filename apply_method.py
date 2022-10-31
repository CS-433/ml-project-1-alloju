from utilities import compute_accuracy, compute_loss_neg_loglikelihood, compute_mse, sigmoid
from paths import  prediction_dir
import os.path as op
import numpy as np
from helpers import create_csv_submission

def apply_method(method,y_tr,x_tr,y_val = np.zeros([10,1]) ,x_val = np.zeros([10,1]), x_te = np.zeros([5,1]), id = np.zeros(5), lambda_ = 0.5, initial_w = None, max_iters = 100, gamma = 0.01, do_predictions = True, validation = True, loss = 'original', logistic = False, separation = False):
    """Apply a given method to the training and validation sets.

    Args:
        method:     method to apply to the data
        y_tr:       training labels
        x_tr:       training features
        y_val:      validation labels
        x_val:      validation features
        x_te:       test features
        id:         index of the labels
        lambda_:    regularisation parameter 
        initial_w:  the initial weight
        max_iters:  the number of iteration maximal 
        gamma:      the learning rate  
        do_predictions: boolean; if true apply the predict function to predict the y label
        validation: boolean; indicates if validation as to be proced
        loss:       type of loss to compute. 'original' for the mse, 'accuracy' to compute the accuracy
        logistic:   boolean; indicates if we have a logistic method 
        separation: boolean; indicates if we use the function for the whole dataset or a fraction of the dataset 

    Returns:
        loss_tr:    training loss
        loss_val:   validation loss
        y_bin:      For separation = True: the predicted labels
    """
    if (initial_w == None):
        initial_w = np.zeros(x_tr.shape[1])
    loss_tr = 0
    loss_val = 0 #To avoid problem if no validation
    if ('reg_logistic_regression' in str(method)):
        w, neg_log_likelihood_tr = method(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
        loss_tr = neg_log_likelihood_tr
    elif ('logistic_regression' in str(method)):
        w, neg_log_likelihood_tr = method(y_tr,x_tr, initial_w, max_iters, gamma)
        loss_tr = neg_log_likelihood_tr
    elif ('mean_squared_error' in str(method) or 'mean_squared_error_sgd' in str(method)):
        w, mse = method(y_tr,x_tr, initial_w, max_iters, gamma)
        loss_tr = mse
    elif ('least_squares' in str(method)):
        w, mse = method(y_tr, x_tr)
        loss_tr = mse
    elif ('ridge_regression' in str(method)):
        w, mse = method(y_tr, x_tr, lambda_)
        loss_tr = mse
    
    #avoid error if no validation set
    loss_val = 0

    if loss == "original":
        if ('logistic_regression' in str(method)):
            loss_tr = neg_log_likelihood_tr
            if validation:
                loss_val = compute_loss_neg_loglikelihood(y_val, x_val, w)
        else:
            loss_tr = mse
            if validation:
                loss_val = compute_mse(y_val, x_val, w)
    elif loss == "accuracy":
        loss_tr = compute_accuracy(y_tr, x_tr, w, logistic)
        loss_val = compute_accuracy(y_val,x_val,w, logistic)

    if do_predictions: 
        if(separation):
            y_bin = predict(method, id, x_te, w, separation)
            return loss_tr, loss_val, y_bin
        predict(method, id, x_te, w)
   
    return loss_tr, loss_val

def predict(method, id, x_te, w, separation = False):
    """Prediction of the y labels 

    Args:
        method: method to apply to the data
        id:         index of the labels
        x_te:       test features
        w:          the initial weights
        separation: boolean; indicates if we use the function for the whole dataset or a fraction of the dataset 

    Returns:
        y_bin:      Only for separation = True; the predicted labels
    """
    y = np.dot(x_te,w)
    # Transform the labels in the right format for the submission
    y_bin = sigmoid(y)
    y_bin[y_bin < 0.5] = -1
    y_bin[y_bin >= 0.5] = 1
    if(separation):
        return y_bin #Only a fraction of the prediction
    path = op.join(prediction_dir, "prediction" + str(method) + ".csv")
    create_csv_submission(id, y_bin, path)


