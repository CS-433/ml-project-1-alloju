from cross_validation import best_single_param_selection, build_k_indices, cross_validation, best_triple_param_selection
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

x = preproc(x) #TODO: decomment

#id, x_te = load_test_data(test_set)
_, x_te, id = load_csv_data(test_set)

x_te = preproc(x_te) #TODO: decomment

#x, _, _ = standardize(x)

#x_te, _, _ = standardize(x_te)

#x_te, x_te_m, xe_te_std = standardize(x_te)
#x_tr, x_val, y_tr, y_val = split_data(x,y,0.8)

#rmse_train, _ = apply_method(im.least_squares, y, x, x_te = x_te, id = id, validation = False)
#print(rmse_train)

#lambda_, cross_rmse_tr_rr, cross_rmse_te_rr = best_single_param_selection(im.ridge_regression, y, x, x_te, id, 10, params = [-1,.0,1], tuned_param = "lambda")
#best_lambda, best_gamma, best_max_iters, best_rmse_val, rmse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y, x, x_te, id, 20, lambdas = [0.5,1,3,5,6,6.5,7,7.5,8,8.5,9,9.5,10,15,50,80], gammas = [0.01, 0.02,0.04,0.05,0.06,0.07,0.1,0.25,0.5,0.75,0.9], maxs_iters = [5,10,15,20,50,75,100,150,200,500])
#best_lambda, best_gamma, best_max_iters, best_rmse_val, rmse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y, x, x_te, id, 20, lambdas = [0.5,1,3,5,6,6.5,7,7.5,8,8.5,9,9.5,10,15,50,80], gammas = [0.01, 0.02,0.04,0.05,0.06,0.07,0.1,0.25,0.5,0.75,0.9], maxs_iters = [5,7,8,9,10,15,20,50,75,100,150,200,500])

#best_lambda, best_gamma, best_max_iters, best_rmse_val, rmse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y, x, x_te, id, 10, lambdas = [0.1,2,5], gammas = [0.1,0.5,0.9], maxs_iters = [6,50])

#best_lambda, best_gamma, best_max_iters, best_rmse_val, rmse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.05,0.1,0.25,0.5,0.75,0.9], maxs_iters = [10,20,50,75,100,150,200])
#TODO: best_lambda, best_gamma, best_max_iters, best_rmse_val, rmse_tr_final = best_triple_param_selection(im.logistic_regression, y, x, x_te, id, 20, lambdas = [0.0], gammas = [0.01, 0.02,0.04,0.05,0.06,0.07,0.1,0.25,0.5,0.75,0.9], maxs_iters = [5,10,15,20,50,75,100, 150])

best_lambda, best_gamma, best_max_iters, best_rmse_val, rmse_tr_final = best_triple_param_selection(im.logistic_regression, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.01,0.05], maxs_iters = [12,20])

#predict(im.ridge_regression, id, x_te, w_tr)
# k_indices = build_k_indices(y, 10, 1)
# print(apply_method(im.ridge_regression, y_tr, x_tr, y_val, x_val, x_te, id, lambda_ = 0.1))
# print(apply_method(im.ridge_regression, y_tr, x_tr, y_val, x_val, x_te, id, lambda_ = 0.5))
# print(cross_validation(im.ridge_regression, y, x, x_te, k_indices, 4, lambda_ = 0.1))
# print(cross_validation(im.ridge_regression, y, x, x_te, k_indices, 4, lambda_ = 0.5))
# print("cross validation on ridge regression: selected lambda = ", lambda_, "cross_rmse_tr_rr = ", cross_rmse_tr_rr, "cross_rmse_te_rr", cross_rmse_te_rr )


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
