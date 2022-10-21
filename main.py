from cross_validation import best_lambda_selection, build_k_indices, cross_validation
from helpers import standardize, load_csv_data
from split_data import split_data
from paths import training_set, test_set
from apply_method import apply_method, predict
import implementations as im
from preprocessing import angle_values, preproc
import numpy as np

#from least_squares import least_squares
#from least_squares_GD import least_squares_GD
#from least_squares_SGD import least_squares_SGD
#from ridge_regression import ridge_regression
#from utilities import compute_mse, compute_rmse


#x,y = load_data(training_set)
y,x,ids = load_csv_data(training_set)
x = preproc(x)
#id, x_te = load_test_data(test_set)
_, x_te, id = load_csv_data(test_set)
x_te = preproc(x_te)
#x_te, x_te_m, xe_te_std = standardize(x_te)
#x_tr, x_val, y_tr, y_val = split_data(x,y,0.8)

#lambda_, cross_rmse_tr_rr, cross_rmse_te_rr = best_lambda_selection(im.ridge_regression, y, x, x_te, id, 10, params = [0.0, 0.00001, 0.01, 0.05, 0.1,0.3,0.5,0.9], tuned_param = "lambda")
lambda_, cross_rmse_tr_rr, cross_rmse_te_rr = best_lambda_selection(im.reg_logistic_regression, y, x, x_te, id, 10, params = [0.1,0.5,1.5,2.5,3,6,6.5,7,8], tuned_param = "lambda")

#predict(im.ridge_regression, id, x_te, w_tr)
# k_indices = build_k_indices(y, 10, 1)
# print(apply_method(im.ridge_regression, y_tr, x_tr, y_val, x_val, x_te, id, lambda_ = 0.1))
# print(apply_method(im.ridge_regression, y_tr, x_tr, y_val, x_val, x_te, id, lambda_ = 0.5))
# print(cross_validation(im.ridge_regression, y, x, x_te, k_indices, 4, lambda_ = 0.1))
# print(cross_validation(im.ridge_regression, y, x, x_te, k_indices, 4, lambda_ = 0.5))
#print("cross validation on ridge regression: selected lambda = ", lambda_, "cross_rmse_tr_rr = ", cross_rmse_tr_rr, "cross_rmse_te_rr", cross_rmse_te_rr )


#rmse_tr_ls, rmse_val_ls = apply_method(im.least_squares, y_tr,x_tr,y_val,x_val, x_te, id)


#print(rmse_tr_ls, rmse_val_ls)

#rmse_tr_lr, rmse_val_lr = apply_method(im.logistic_regression, y_tr,x_tr,y_val,x_val, x_te, id, gamma = 0.05) #y, tx, initial_w, max_iters, gamma
# rmse_tr_mss, rmse_val_mss = apply_method(im.mean_squared_error_sgd, y_tr,x_tr,y_val,x_val, x_te, id)
# rmse_tr_msg, rmse_val_msg = apply_method(im.mean_squared_error_gd, y_tr,x_tr,y_val,x_val, x_te, id)
# rmse_tr_rr, rmse_val_rr = apply_method(im.ridge_regression, y_tr,x_tr,y_val,x_val, x_te, id)
#rmse_tr_rlr, rmse_val_rlr = apply_method(im.reg_logistic_regression, y_tr, x_tr, y_val, x_val, x_te, id, lambda_=10)
#print("reg logistic regression rmse: ", rmse_tr_rlr, rmse_val_rlr)
# print("least squares rmse: ", rmse_tr_ls, rmse_val_ls)
#print("logistic regression rmse: ", rmse_tr_lr, rmse_val_lr)
# print("ridge regression: ", rmse_tr_rr, rmse_val_rr)
# print("mean squared SGD: ", rmse_tr_mss, rmse_val_mss)
# print("mean squared GD: ", rmse_tr_msg, rmse_val_msg)


def test(a,b,c):
    """_summary_

    Args:
        a (_type_): _description_
        b (_type_): _description_
        c (_type_): _description_

    Returns:
        _type_: _description_
    """
    #autodocstring
    info = 0
    return info
