from sys import implementation
from utilities import compute_accuracy, compute_loss_neg_loglikelihood, compute_mse,compute_rmse, sigmoid
from paths import  prediction_dir
import os.path as op
import numpy as np
from helpers import create_csv_submission
from preprocessing import replace_class, class_separation, preproc_train, preproc_test
from split_data import split_data

def apply_method(method,y_tr,x_tr,y_val = np.zeros([10,1]) ,x_val = np.zeros([10,1]), x_te = np.zeros([5,1]), id = np.zeros(5), lambda_ = 0.5, initial_w = None, max_iters = 100, gamma = 0.01, do_predictions = True, validation = True, loss = 'original', logistic = False, separation = False):
    """Apply a given method to the training and validation sets.

    Args:
        method: method to apply to the data
        y_tr: training labels
        x_tr: training features
        y_val: validation labels
        x_val: validation features
        x_te: 
        separation: 

    Returns:
        rmse_tr: training rmse
        rmse_val: validation rmse
        y_bin: The prediction for the input data

    """

    # TODO: if blablabla in file name 
    #une manière plus élégante de faire maybe ?:
    #import foo
    #bar = getattr(foo, 'bar')
    #result = bar()

    if (initial_w == None):
        initial_w = np.zeros(x_tr.shape[1])
    loss_tr = 0
    loss_val = 0 #to avoid problem if no validation
    if ('reg_logistic_regression' in str(method)):
        w, neg_log_likelihood_tr = method(y_tr, x_tr, lambda_, initial_w, max_iters, gamma)
        loss_tr = neg_log_likelihood_tr
        #mse = compute_mse(y_tr, x_tr, w)
    elif ('logistic_regression' in str(method)):
        w, neg_log_likelihood_tr = method(y_tr,x_tr, initial_w, max_iters, gamma)
        loss_tr = neg_log_likelihood_tr
        #mse = compute_mse(y_tr, x_tr, w)
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
    acc_val = 0
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
        loss_tr = compute_accuracy(y_tr, x_tr, w)
        loss_val = compute_accuracy(y_val,x_val,w)
    #TODO: add other possibilities of calculations !
    #if validation:
    # acc_val = compute_accuracy(y_val,x_val,w)
    if do_predictions: # and x_te == None:
        if(separation):
            y_bin = predict(method, id, x_te, w, separation)
        else: 
            predict(method, id, x_te, w)
    #acc_train = compute_accuracy(y_tr, x_tr, w)
     
    if(separation):
        return loss_tr, loss_val, y_bin
    else:
        #return acc_train, acc_val
        return loss_tr, loss_val

def predict(method, id, x_te, w, separation = False):
    """_summary_

    Args:
        method (_type_): _description_
        id (_type_): _description_
        x_te (_type_): _description_
        separation: 
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

def apply_separation_method(method,y_tr,x_tr, id_tr, x_te = np.zeros([5,1]), id_te = np.zeros(5), lambda_ = 0.5, initial_w = None, max_iters = 100, gamma = 0.01, do_corr = False, do_pca = False, do_poly = False, percentage = 95, logistic = False):
# method,y_tr,x_tr,y_val = np.zeros([10,1]) ,x_val = np.zeros([10,1]), x_te = np.zeros([5,1]), id = np.zeros(5), lambda_ = 0.5, initial_w = None, max_iters = 100, gamma = 0.01, do_predictions = True, validation = True, loss = 'original', separation = False
    x_tr, title_tr = replace_class(x_tr, title_tr)
    xs, ys_, ids_ = class_separation(x_tr, title_tr, id_tr, y_tr)
    x_te, title_te = replace_class(x_te, title_te)
    xs_te, _, ids_te = class_separation(x_te, title_te, id_te, _)

    mse_val_s = []
    ys = []
    ids = []
    acc_trains = []
    acc_vals = []
    for i in range(len(xs)):
        xs[i], x_mean, x_std, ind, projection_matrix = preproc_train(xs[i], title_tr, percentage = percentage, do_corr = do_corr, do_pca = do_pca, do_poly = do_poly) 
        xs_te[i] = preproc_test(xs_te[i], title_te, x_mean, x_std, projection_matrix, ind, do_corr = do_corr, do_pca = do_pca, do_poly = do_poly)
        x_tr_sep, x_val_sep, y_tr_sep, y_val_sep = split_data(xs[i],ys_[i],0.8)
        #Do cross val:
        mse_train, mse_val, y_bin = apply_method(method, x_tr_sep, y_tr_sep, x_val_sep, y_val_sep, lambda_ = lambda_, initial_w = initial_w, max_iters = max_iters, gamma = gamma, loss = "original", validation = True,do_predictions= False, separation = True, logistic = logistic)
        #prepare predictions
        mse_train, _, y_bin = apply_method(method, ys_[i], xs[i], x_te = xs_te[i], id = ids_te[i], lambda_ = lambda_, initial_w = initial_w, max_iters = max_iters, gamma = gamma, loss = "original", validation = False, separation = True, do_predictions = True, logistic = logistic)
        #calculate accuracies
        acc_train, acc_val = apply_method(method, y_tr_sep, x_tr_sep, y_val = y_val_sep, x_val = x_val_sep, lambda_ = lambda_, initial_w = initial_w, max_iters = max_iters, gamma = gamma, do_predictions = False, validation = True, loss = "accuracy", logistic = logistic, separation = False)
        print("Accuracy:", acc_train, acc_val)
        ratio = xs[i].shape[0]/len(x_tr)
        mse_val_s.append(mse_val*ratio)
        acc_trains.append(acc_train*ratio)
        acc_vals.append( acc_val*ratio)
        ys = np.concatenate((ys, y_bin))
        ids = np.concatenate((ids,ids_te[i]))

    ids = np.squeeze(ids)
    ids_ys = np.array([ids.T, ys.T], np.int32)
    #Sort in the same order than in the load_csv_data input
    index = np.argsort(ids_ys[0])
    ids_ys[1] = ids_ys[1][index]
    ids_ys[0] = ids_ys[0][index]
    joining_prediction(method, ids_ys[0], ids_ys[1])
    print(np.sum(mse_val_s))
    print('Accuracy:', np.sum(acc_trains), np.sum(acc_vals))
