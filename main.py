from helpers import standardize
from split_data import load_data, split_data, load_test_data
from paths import training_set, test_set
from apply_method import apply_method
import implementations as im
from preprocessing import angle_values, normalize, preproc

#from least_squares import least_squares
#from least_squares_GD import least_squares_GD
#from least_squares_SGD import least_squares_SGD
#from ridge_regression import ridge_regression
#from utilities import compute_mse, compute_rmse


x,y = load_data(training_set)
x = preproc(x)
id, x_te = load_test_data(test_set) 
x_te, x_te_m, xe_te_std = standardize(x_te)
x_te = angle_values(x_te)

x_tr, x_val, y_tr, y_val = split_data(x,y,0.8)
rmse_tr_ls, rmse_val_ls = apply_method(im.least_squares, y_tr,x_tr,y_val,x_val, x_te, id)
rmse_tr_ls, rmse_val_ls = apply_method(im.logistic_regression, y_tr,x_tr,y_val,x_val, x_te, id) #y, tx, initial_w, max_iters, gamma
#rmse_tr_ls, rmse_val_ls = apply_method(im.mean_squared_SGD, y_tr,x_tr,y_val,x_val, x_te, id)
#rmse_tr_ls, rmse_val_ls = apply_method(im.mean_squared_GD, y_tr,x_tr,y_val,x_val, x_te, id)
#rmse_tr_ls, rmse_val_ls = apply_method(im.ridge_regression, y_tr,x_tr,y_val,x_val, x_te, id)

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
