from cross_validation import best_single_param_selection, build_k_indices, cross_validation, best_triple_param_selection
from helpers import standardize, load_csv_data, load_csv_title
from split_data import split_data
from paths import training_set, test_set
from apply_method import apply_method, predict
import implementations as im
from preprocessing import angle_values, preproc_test, preproc_train, to_0_1
import numpy as np

#from least_squares import least_squares
#from least_squares_GD import least_squares_GD
#from least_squares_SGD import least_squares_SGD
#from ridge_regression import ridge_regression
#from utilities import compute_mse, compute_mse


#x,y = load_data(training_set)

# TODO: décommenter
y,x,ids = load_csv_data(training_set)
title = load_csv_title(training_set)

x, x_mean, x_std, ind, projection_matrix = preproc_train(x, title, do_corr = False, do_pca = False) #TODO: decomment

_, x_te, id = load_csv_data(test_set)
title = load_csv_title(test_set)

x_te = preproc_test(x_te, title, x_mean, x_std, projection_matrix, ind, do_corr = False, do_pca = False) #TODO: decomment

y = to_0_1(y)

# TODO: stop décommenter


# LEAST SQUARES

#For validation:
#mse_train, mse_val = apply_method(im.least_squares, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = id, validation = True)
#print(mse_train, mse_val)

# For prediction: 
mse_train, _ = apply_method(im.least_squares, y, x, x_te = x_te, id = id, validation = False)
print(mse_train)

# RIDGE REGRESSION

#best_lambda_, cross_mse_tr_rr, cross_mse_val_rr = best_single_param_selection(im.ridge_regression, y, x, x_te, id, 10, params = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], tuned_param = "lambda")


# LOG REG

#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.05,0.06,0.07, 0.1], maxs_iters = [500, 1000, 1200])

# REG LOG REG

#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y, x, x_te, id, 20, lambdas = [0.5,1,3,5,6,6.5,7,7.5,8,8.5,9,9.5,10,15,50,80], gammas = [0.01, 0.02,0.04,0.05,0.06,0.07,0.1,0.25,0.5,0.75,0.9], maxs_iters = [5,10,15,20,50,75,100,150,200,500])
#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y, x, x_te, id, 20, lambdas = [0.5,1,3,5,6,6.5,7,7.5,8,8.5,9,9.5,10,15,50,80], gammas = [0.01, 0.02,0.04,0.05,0.06,0.07,0.1,0.25,0.5,0.75,0.9], maxs_iters = [5,7,8,9,10,15,20,50,75,100,150,200,500])

#y,x,ids = load_csv_data(training_set)
#title = load_csv_title(training_set)

#x, x_mean, x_std, ind, projection_matrix = preproc_train(x, title, do_corr = True, do_pca = False) #TODO: decomment

# #id, x_te = load_test_data(test_set)
# _, x_te, id = load_csv_data(test_set)

#x_te = preproc_test(x_te, title, x_mean, x_std, projection_matrix, ind, do_corr = True, do_pca = False) #TODO: decomment

# y = to_0_1(y)

#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y, x, x_te, id, 10, lambdas = [0.001,0.01, 0.1], gammas = [0.1,0.5,0.9], maxs_iters = [6,50, 500, 1000])

#GRADIENT DESCENT:

#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.001, 0.02, 0.03,0.04,0.05,0.055,0.06,0.065,0.07, 0.1], maxs_iters = [50,100,200,300,400,500,600,700, 800, 900, 1000, 1200, 1500])

#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.001, 0.01, 0.1], maxs_iters = [50,100, 1000])
#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.04, 0.045,0.05,0.055,0.06,0.065], maxs_iters = [1000, 1200, 1400, ])

#gamma, best_mse_val, mse_tr_final = best_single_param_selection(im.mean_squared_error_gd, y,x, x_te, id, 10, params = [0.045,0.05,0.055,0.1], tuned_param = "gamma", lambda_ = 0, max_iters= 1500)
#max_iters, best_mse_val, mse_tr_final = best_single_param_selection(im.mean_squared_error_gd, y,x, x_te, id, 10, params = [100,500,1200,1250,1300], tuned_param = "max_iters", lambda_ = 0, gamma= 0.06)
#TODO: best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.logistic_regression, y, x, x_te, id, 20, lambdas = [0.0], gammas = [0.01, 0.02,0.04,0.05,0.06,0.07,0.1,0.25,0.5,0.75,0.9], maxs_iters = [5,10,15,20,50,75,100, 150])

#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.logistic_regression, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.01,0.05], maxs_iters = [12,20])

# SUBGRADIENT DESCENT:

#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.001, 0.03,0.05,0.06,0.07, 0.1], maxs_iters = [50,100,500,1000])
#gamma, best_mse_val, mse_tr_final = best_single_param_selection(im.mean_squared_error_sgd, y,x, x_te, id, 10, params = [0.01,0.02,0.06,0.1, 0.2, 0.5], tuned_param = "gamma", lambda_ = 0, max_iters= 50)
#max_iters, best_mse_val, mse_tr_final = best_single_param_selection(im.mean_squared_error_gd, y,x, x_te, id, 10, params = [100,500,1200,1250,1300], tuned_param = "max_iters", lambda_ = 0, gamma= 0.06)


#predict(im.ridge_regression, id, x_te, w_tr)
# k_indices = build_k_indices(y, 10, 1)
# print(apply_method(im.ridge_regression, y_tr, x_tr, y_val, x_val, x_te, id, lambda_ = 0.1))
# print(apply_method(im.ridge_regression, y_tr, x_tr, y_val, x_val, x_te, id, lambda_ = 0.5))
# print(cross_validation(im.ridge_regression, y, x, x_te, k_indices, 4, lambda_ = 0.1))
# print(cross_validation(im.ridge_regression, y, x, x_te, k_indices, 4, lambda_ = 0.5))
# print("cross validation on ridge regression: selected lambda = ", lambda_, "cross_mse_tr_rr = ", cross_mse_tr_rr, "cross_mse_te_rr", cross_mse_te_rr )


#mse_tr_ls, mse_val_ls = apply_method(im.least_squares, y_tr,x_tr,y_val,x_val, x_te, id)


#print(mse_tr_ls, mse_val_ls)

#mse_tr_lr, mse_val_lr = apply_method(im.logistic_regression, y_tr,x_tr,y_val,x_val, x_te, id, gamma = 0.05) #y, tx, initial_w, max_iters, gamma
# mse_tr_mss, mse_val_mss = apply_method(im.mean_squared_error_sgd, y_tr,x_tr,y_val,x_val, x_te, id)
# mse_tr_msg, mse_val_msg = apply_method(im.mean_squared_error_gd, y_tr,x_tr,y_val,x_val, x_te, id)
# mse_tr_rr, mse_val_rr = apply_method(im.ridge_regression, y_tr,x_tr,y_val,x_val, x_te, id)
#mse_tr_rlr, mse_val_rlr = apply_method(im.reg_logistic_regression, y_tr, x_tr, y_val, x_val, x_te, id, lambda_=10)
#print("reg logistic regression mse: ", mse_tr_rlr, mse_val_rlr)
# print("least squares mse: ", mse_tr_ls, mse_val_ls)
#print("logistic regression mse: ", mse_tr_lr, mse_val_lr)
# print("ridge regression: ", mse_tr_rr, mse_val_rr)
# print("mean squared SGD: ", mse_tr_mss, mse_val_mss)
# print("mean squared GD: ", mse_tr_msg, mse_val_msg)
