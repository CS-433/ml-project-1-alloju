from split_data import load_data, split_data, load_test_data
from paths import training_set, test_set
from least_squares import least_squares
from least_squares_GD import least_squares_GD
from least_squares_SGD import least_squares_SGD
from ridge_regression import ridge_regression
from utilities import compute_mse, compute_rmse
from apply_method import apply_method

x,y = load_data(training_set)
id, x_te = load_test_data(test_set) 
x_tr, x_val, y_tr, y_val = split_data(x,y,0.8)
rmse_tr_ls, rmse_val_ls = apply_method(least_squares, y_tr,x_tr,y_val,x_val, x_te, id)
#rmse_tr_ls, rmse_val_ls = apply_method(least_squares_SGD, y_tr,x_tr,y_val,x_val, x_te, id)
#rmse_tr_ls, rmse_val_ls = apply_method(least_squares_GD, y_tr,x_tr,y_val,x_val, x_te, id)
#rmse_tr_ls, rmse_val_ls = apply_method(ridge_regression, y_tr,x_tr,y_val,x_val, x_te, id)
