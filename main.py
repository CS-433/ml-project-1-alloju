from cross_validation import best_degree_selection, best_single_param_selection, build_k_indices, cross_validation, best_triple_param_selection, joining_prediction, apply_separation_method
from helpers import standardize, load_csv_data, load_csv_title
from split_data import split_data
from paths import training_set, test_set
from apply_method import apply_method, predict
import implementations as im
from preprocessing import angle_values, preproc_test, preproc_train, to_0_1, class_separation, replace_class
import numpy as np

#from least_squares import least_squares
#from least_squares_GD import least_squares_GD
#from least_squares_SGD import least_squares_SGD
#from ridge_regression import ridge_regression
#from utilities import compute_mse, compute_mse


# TODO: décommenter
y,x,id_tr = load_csv_data(training_set)
title_tr = load_csv_title(training_set)
#x, title_tr = replace_class(x, title_tr)
y_logistic = to_0_1(y)
#xs, ys_, ids_ = class_separation(x, title_tr, id_tr, y)

_, x_te, id_te = load_csv_data(test_set)
title_te = load_csv_title(test_set)
#x_te, title_te = replace_class(x_te, title_te)
#xs_te, _s, ids_te = class_separation(x_te, title_te, id_te, _)

#apply_separation_method(method = im.least_squares , y_tr = y ,x_tr = x, id_tr = id_tr, title_tr = title_tr, y_te = _, x_te = x_te, id_te = id_te, title_te = title_te, k_fold = 10, lambdas_ = [0.5], initial_w = None, max_iters = [100], gammas = [0.01], do_corr = False, do_pca = False, percentage = 95, logistic = False, verbose = False) #do_poly = False
#apply_separation_method(method = im.mean_squared_error_gd , y_tr = y ,x_tr = x, id_tr = id_tr, title_tr = title_tr, y_te = _, x_te = x_te, id_te = id_te, title_te = title_te, k_fold = 10, lambdas_ = [1e-3, 0.1], initial_w = None, max_iters = [30, 100], gammas = [2e-3, 0.2], do_corr = False, do_pca = False, percentage = 95, logistic = False, verbose = True) #do_poly = False
#apply_separation_method(method = im.mean_squared_error_sgd , y_tr = y ,x_tr = x, id_tr = id_tr, title_tr = title_tr, y_te = _, x_te = x_te, id_te = id_te, title_te = title_te, k_fold = 10, lambdas_ = [1e-3, 0.1], initial_w = None, max_iters = [30, 100], gammas = [2e-3, 0.2], do_corr = False, do_pca = False, percentage = 95, logistic = False, verbose = True) #do_poly = False
apply_separation_method(method = im.ridge_regression , y_tr = y ,x_tr = x, id_tr = id_tr, title_tr = title_tr, y_te = _, x_te = x_te, id_te = id_te, title_te = title_te, k_fold = 10, lambdas_ = [123e-8, 1e-8, 1e-10, 2e-7, 3e-6, 1e-6, 1e-5 , 1e-4, 2e-4, 3e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,  2e-3, 8e-3, 1e-3, 5e-2, 1e-2, 7e-2, 1e-1, 2e-1, 4e-2, 5e-1, 8e-2, 9e-2, 0.0, 1, 8], initial_w = None, max_iters = [1], gammas = [0.0], do_corr = False, do_pca = False, percentage = 95, logistic = False, verbose = True) #do_poly = False
#apply_separation_method(method = im.logistic_regression , y_tr = y_logistic ,x_tr = x, id_tr = id_tr, title_tr = title_tr, y_te = _, x_te = x_te, id_te = id_te, title_te = title_te, k_fold = 10, lambdas_ = [1e-3, 0.1], initial_w = None, max_iters = [30, 100], gammas = [2e-3, 0.2], do_corr = False, do_pca = False, percentage = 95, logistic = True, verbose = True) #do_poly = False
#apply_separation_method(method = im.reg_logistic_regression , y_tr = y_logistic ,x_tr = x, id_tr = id_tr, title_tr = title_tr, y_te = _, x_te = x_te, id_te = id_te, title_te = title_te, k_fold = 10, lambdas_ = [1e-3, 0.1, 1e-5, 1e-4, 0.03, 0.23], initial_w = None, max_iters = [100, 400], gammas = [2e-3, 1e-7, 1e-3, 0.1, 0.2], do_corr = False, do_pca = False, percentage = 95, logistic = True, verbose = True) #do_poly = False


"""
mse_train_s = []
ys = []
ids = []
acc_trains = []
acc_vals = []
method = im.least_squares
for i in range(len(xs)):
    xs[i], x_mean, x_std, ind, projection_matrix = preproc_train(xs[i], title_tr, percentage = 95, do_corr = False, do_pca = False) 
    xs_te[i] = preproc_test(xs_te[i], title_te, x_mean, x_std, projection_matrix, ind, do_corr = False, do_pca = False)
    mse_train, _, y_bin = apply_method(method, ys_[i], xs[i], x_te = xs_te[i], id = ids_te[i], validation = False, separation = True)
    x_tr, x_val, y_tr, y_val = split_data(xs[i],ys_[i],0.8)
    ## SI validation par mini groupe Juste le ids_ pas dimension donc je sais pas lequel tu prenais d'habitude 
    acc_train, acc_val = apply_method(method, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = ids_, do_predictions = False, validation = True, loss = "accuracy")
    print("Accuracy:", acc_train, acc_val)
    ## jusque la 
    ratio = xs[i].shape[0]/len(x)
    mse_train_s.append(mse_train*ratio)
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
print(np.sum(mse_train_s))
print('Accuracy:', np.sum(acc_trains), np.sum(acc_vals))
"""

"""
mse_train_s = []
ys = []
ids = []
acc_trains = []
acc_vals = []
method = im.ridge_regression
lamb = 123e-8
for i in range(len(xs)):
    xs[i], x_mean, x_std, ind, projection_matrix = preproc_train(xs[i], title_tr, percentage = 95, do_corr = False, do_pca = False) 
    xs_te[i] = preproc_test(xs_te[i], title_te, x_mean, x_std, projection_matrix, ind, do_corr = False, do_pca = False)
    mse_train, _, y_bin = apply_method(method, ys_[i], xs[i], x_te = xs_te[i], id = ids_te[i], lambda_=lamb, validation = False, separation = True)
    x_tr, x_val, y_tr, y_val = split_data(xs[i],ys_[i],0.8)
    ## SI validation par mini groupe Juste le ids_ pas dimension donc je sais pas lequel tu prenais d'habitude 
    acc_train, acc_val = apply_method(method, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = ids_, lambda_=lamb, do_predictions = False, validation = True, loss = "accuracy")
    print("Accuracy individual:", acc_train, acc_val)
    ## jusque la 
    ratio = xs[i].shape[0]/len(x)
    mse_train_s.append(mse_train*ratio)
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
print(np.sum(mse_train_s))
print('Accuracy:', np.sum(acc_trains), np.sum(acc_vals))
"""
"""
mse_train_s = []
ys = []
ids = []
acc_trains = []
acc_vals = []
method = im.mean_squared_error_gd
max_i = 1000
gam = 0.05
for i in range(len(xs)):
    xs[i], x_mean, x_std, ind, projection_matrix = preproc_train(xs[i], title_tr, percentage = 95, do_corr = False, do_pca = False) 
    xs_te[i] = preproc_test(xs_te[i], title_te, x_mean, x_std, projection_matrix, ind, do_corr = False, do_pca = False)
    mse_train, _, y_bin = apply_method(method, ys_[i], xs[i], x_te = xs_te[i], id = ids_te[i], max_iters = max_i, gamma = gam, validation = False, separation = True)
    x_tr, x_val, y_tr, y_val = split_data(xs[i],ys_[i],0.8)
    ## SI validation par mini groupe Juste le ids_ pas dimension donc je sais pas lequel tu prenais d'habitude 
    acc_train, acc_val = apply_method(method, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = ids_, max_iters = max_i, gamma = gam,  do_predictions = False, validation = True, loss = "accuracy")
    print("Accuracy individual:", acc_train, acc_val)
    ## jusque la 
    ratio = xs[i].shape[0]/len(x)
    mse_train_s.append(mse_train*ratio)
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
print(np.sum(mse_train_s))
print('Accuracy:', np.sum(acc_trains), np.sum(acc_vals))
"""
"""
mse_train_s = []
ys = []
ids = []
acc_trains = []
acc_vals = []
method = im.mean_squared_error_sgd
max_i = 2000
gam = 1e-4
lamb = 0.0001
for i in range(len(xs)):
    xs[i], x_mean, x_std, ind, projection_matrix = preproc_train(xs[i], title_tr, percentage = 95, do_corr = False, do_pca = False) 
    xs_te[i] = preproc_test(xs_te[i], title_te, x_mean, x_std, projection_matrix, ind, do_corr = False, do_pca = False)
    mse_train, _, y_bin = apply_method(method, ys_[i], xs[i], x_te = xs_te[i], id = ids_te[i], max_iters = max_i, gamma = gam, lambda_=lamb, validation = False, separation = True)
    x_tr, x_val, y_tr, y_val = split_data(xs[i],ys_[i],0.8)
    ## SI validation par mini groupe Juste le ids_ pas dimension donc je sais pas lequel tu prenais d'habitude 
    acc_train, acc_val = apply_method(method, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = ids_, max_iters = max_i, gamma = gam, lambda_=lamb, do_predictions = False, validation = True, loss = "accuracy")
    print("Accuracy individual:", acc_train, acc_val)
    ## jusque la 
    ratio = xs[i].shape[0]/len(x)
    mse_train_s.append(mse_train*ratio)
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
print(np.sum(mse_train_s))
print('Accuracy:', np.sum(acc_trains), np.sum(acc_vals))
"""
"""
mse_train_s = []
ys = []
ids = []
acc_trains = []
acc_vals = []
method = im.logistic_regression
max_i = 400
gam = 0.2
for i in range(len(xs)):
    xs[i], x_mean, x_std, ind, projection_matrix = preproc_train(xs[i], title_tr, percentage = 95, do_corr = False, do_pca = False) 
    xs_te[i] = preproc_test(xs_te[i], title_te, x_mean, x_std, projection_matrix, ind, do_corr = False, do_pca = False)
    mse_train, _, y_bin = apply_method(method, ys_[i], xs[i], x_te = xs_te[i], id = ids_te[i], max_iters = max_i, gamma = gam, validation = False, separation = True)
    x_tr, x_val, y_tr, y_val = split_data(xs[i],ys_[i],0.8)
    ## SI validation par mini groupe Juste le ids_ pas dimension donc je sais pas lequel tu prenais d'habitude 
    acc_train, acc_val = apply_method(method, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = ids_, max_iters = max_i, gamma = gam,  do_predictions = False, validation = True, loss = "accuracy")
    print("Accuracy individual:", acc_train, acc_val)
    ## jusque la 
    ratio = xs[i].shape[0]/len(x)
    mse_train_s.append(mse_train*ratio)
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
print(np.sum(mse_train_s))
print('Accuracy:', np.sum(acc_trains), np.sum(acc_vals))
"""
"""
mse_train_s = []
ys = []
ids = []
acc_trains = []
acc_vals = []
method = im.reg_logistic_regression
max_i = 1200
gam = 0.05
lamb = 0.001
for i in range(len(xs)):
    xs[i], x_mean, x_std, ind, projection_matrix = preproc_train(xs[i], title_tr, percentage = 95, do_corr = False, do_pca = False) 
    xs_te[i] = preproc_test(xs_te[i], title_te, x_mean, x_std, projection_matrix, ind, do_corr = False, do_pca = False)
    mse_train, _, y_bin = apply_method(method, ys_[i], xs[i], x_te = xs_te[i], id = ids_te[i], max_iters = max_i, gamma = gam, lambda_=lamb, validation = False, separation = True)
    x_tr, x_val, y_tr, y_val = split_data(xs[i],ys_[i],0.8)
    ## SI validation par mini groupe Juste le ids_ pas dimension donc je sais pas lequel tu prenais d'habitude 
    acc_train, acc_val = apply_method(method, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = ids_, max_iters = max_i, gamma = gam, lambda_=lamb, do_predictions = False, validation = True, loss = "accuracy")
    print("Accuracy individual:", acc_train, acc_val)
    ## jusque la 
    ratio = xs[i].shape[0]/len(x)
    mse_train_s.append(mse_train*ratio)
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
print(np.sum(mse_train_s))
print('Accuracy:', np.sum(acc_trains), np.sum(acc_vals))
"""

#For validation: A Mettre dans la boucle ! x,y = xs[i], ys[i]
#x_tr, x_val, y_tr, y_val = split_data(x,y,0.8)
# mse_train, mse_val = apply_method(im.least_squares, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = id, validation = True)
# print("MSE: ", mse_train, mse_val)
# acc_train, acc_val = apply_method(im.least_squares, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = id, validation = True, loss = "accuracy")
# print("Accuracy:", acc_train, acc_val)

#x,y = load_data(training_set)

# TODO: décommenter

y,x,ids = load_csv_data(training_set)
title = load_csv_title(training_set)
chosen_degree = 5

print("nothing:")
x, x_mean, x_std, ind, projection_matrix = preproc_train(x, title, do_corr = False, do_pca = False, do_poly = False) #TODO: decomment

_, x_te, id = load_csv_data(test_set)
title = load_csv_title(test_set)

x_te = preproc_test(x_te, title, x_mean, x_std, projection_matrix, ind, do_corr = False, do_pca = False, do_poly = False) #TODO: decomment
y_logistic = to_0_1(y)

x_tr, x_val, y_tr, y_val = split_data(x,y_logistic,0.8)
print("Data have been preprocessed")

mse_train, mse_val = apply_method(im.least_squares, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = id, validation = True, logistic = True)
print("MSE: ", mse_train, mse_val)
acc_train, acc_val = apply_method(im.least_squares, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = id, validation = True, loss = "accuracy", logistic = True)
print("Accuracy:", acc_train, acc_val)

print("do_corr")
y,x,ids = load_csv_data(training_set)
title = load_csv_title(training_set)
x, x_mean, x_std, ind, projection_matrix = preproc_train(x, title, do_corr = True, do_pca = False, do_poly = False) #TODO: decomment

_, x_te, id = load_csv_data(test_set)
title = load_csv_title(test_set)

x_te = preproc_test(x_te, title, x_mean, x_std, projection_matrix, ind, do_corr = True, do_pca = False, do_poly = False) #TODO: decomment
y_logistic = to_0_1(y)

# TODO: stop décommenter

#x_tr, x_val, y_tr, y_val = split_data(x,y,0.8)

# LEAST SQUARES

#For validation:
# mse_train, mse_val = apply_method(im.least_squares, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = id, validation = True)
# print("MSE: ", mse_train, mse_val)
# acc_train, acc_val = apply_method(im.least_squares, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = id, validation = True, loss = "accuracy")
# print("Accuracy:", acc_train, acc_val)


# For prediction: 
#mse_train, _ = apply_method(im.least_squares, y, x, x_te = x_te, id = id, validation = False)
#print(mse_train)

# RIDGE REGRESSION

#best_lambda_, cross_mse_tr_rr, cross_mse_val_rr = best_single_param_selection(im.ridge_regression, y, x, x_te, id, 10, params = [0.0, 1e-6, 1e-5, 1e-4, 2e-4, 3e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 1e-2, 1e-1], tuned_param = "lambda")


# LOG REG

#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [1e-6, 1e-5, 1e-4, 1e-3, 0.03, 0.04, 0.05,0.06,0.07, 0.09], maxs_iters = [50, 100, 250, 300, 350, 450, 500, 550, 650, 700, 1000, 1100, 1200, 1500])


#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.logistic_regression, y, x, x_te, id, 10, lambdas = [0.0], gammas = [1e-3,1e-2, 1e-1], maxs_iters = [50, 500,1000])

#best_max_iters, best_loss_tr, best_loss_val = best_single_param_selection(im.logistic_regression, y,x,x_te, id, 10, params = [20,50,100,350,400,450,500], gamma = 0.055, tuned_param = "max_iters")
#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [1e-6, 1e-5, 1e-4, 1e-3, 0.05,0.07, 0.1], maxs_iters = [50, 100, 500, 1000])
#TODO: run apply method with : lambda =  0.0 max_iters =  500 gamma =  0.05 loss_val =  0.3747762495867381

# REG LOG REG

print("reg log reg PCA 95%")

best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y_logistic, x, x_te, id, 10, lambdas = [1e-6, 1e-5 , 1e-4], gammas = [5e-2, 1e-1, 5e-1], maxs_iters = [1200])

x, x_mean, x_std, ind, projection_matrix = preproc_train(x, title, percentage = 80, do_corr = False, do_pca = True) #TODO: decomment
x_te = preproc_test(x_te, title, x_mean, x_std, projection_matrix, ind, do_corr = False, do_pca = True) #TODO: decomment

print("reg log reg PCA 80%")

best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y_logistic, x, x_te, id, 10, lambdas = [1e-6, 1e-5 , 1e-4], gammas = [5e-2, 1e-1, 5e-1], maxs_iters = [1200])

print("reg log reg PCA 99%")

x, x_mean, x_std, ind, projection_matrix = preproc_train(x, title, percentage = 99, do_corr = False, do_pca = True) #TODO: decomment
x_te = preproc_test(x_te, title, x_mean, x_std, projection_matrix, ind, do_corr = False, do_pca = True) #TODO: decomment

best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y_logistic, x, x_te, id, 10, lambdas = [1e-6, 1e-5 , 1e-4], gammas = [5e-2, 1e-1, 5e-1], maxs_iters = [1200])


#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y_logistic, x, x_te, id, 20, lambdas = [0.5,1,3,5,6,6.5,7,7.5,8,8.5,9,9.5,10,15,50,80], gammas = [0.01, 0.02,0.04,0.05,0.06,0.07,0.1,0.25,0.5,0.75,0.9], maxs_iters = [5,10,15,20,50,75,100,150,200,500])
#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y_logistic, x, x_te, id, 10, lambdas = [1e-5 , 1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1], gammas = [1e-5, 1e-4, 5e-3, 1e-3, 9e-2, 7e-2, 5e-2, 3e-2, 1e-2, 1e-1, 5e-1], maxs_iters = [5,7,8,9,10,15,20,50,75,100,150,200,500, 1000, 1200])

#worse :( #best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y_logistic, x, x_te, id, 10, lambdas = [5e-5 , 1e-4, 5e-4], gammas = [3e-1, 5e-1, 7e-1], maxs_iters = [400,500,600])


#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.reg_logistic_regression, y, x, x_te, id, 20, lambdas = [1e-2], gammas = [0.01,0.05], maxs_iters = [5])

#y,x,ids = load_csv_data(training_set)
#title = load_csv_title(training_set)

#x, x_mean, x_std, ind, projection_matrix = preproc_train(x, title, do_corr = True, do_pca = False) #TODO: decomment

# #id, x_te = load_test_data(test_set)
# _, x_te, id = load_csv_data(test_set)

#x_te = preproc_test(x_te, title, x_mean, x_std, projection_matrix, ind, do_corr = True, do_pca = False) #TODO: decomment

# y = to_0_1(y)

#best_lambda, best_gamma, best_max_iters, mse_tr_final, best_mse_val = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5], maxs_iters = [100, 1000, 1200])

#GRADIENT DESCENT:
#print("gradient descent")
#TODO: run : best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [1e-5,1e-4,0.001, 0.02, 0.03,0.04,0.05,0.055,0.06,0.065,0.07, 0.1], maxs_iters = [50,100,200,300,400,500,600,700, 800, 900, 1000, 1100,1200, 1300,1500])

#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.04, 0.045, 0.055], maxs_iters = [900, 1000])
#print ("GD, with nothing")
#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.04,0.05, 0.06,0.07], maxs_iters = [400, 500, 600, 750, 1000], verbose = False)
#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.04, 0.045,0.05,0.055,0.06,0.065], maxs_iters = [1000, 1200, 1400, ])

#gamma, best_mse_tr, best_mse_val = best_single_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, params = [1e-4,1e-3,1e-2,0.05,0.07,0.09,0.1,0.2,0.3, 5e-1], tuned_param = "gamma", max_iters= 1000)
#gamma, best_mse_val, mse_tr_final = best_single_param_selection(im.mean_squared_error_gd, y,x, x_te, id, 10, params = [0.045,0.05,0.055,0.1], tuned_param = "gamma", lambda_ = 0, max_iters= 1500)
#max_iters, best_mse_val, mse_tr_final = best_single_param_selection(im.mean_squared_error_gd, y,x, x_te, id, 10, params = [100,500,1200,1250,1300], tuned_param = "max_iters", lambda_ = 0, gamma= 0.06)
#TODO: best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.logistic_regression, y, x, x_te, id, 20, lambdas = [0.0], gammas = [0.01, 0.02,0.04,0.05,0.06,0.07,0.1,0.25,0.5,0.75,0.9], maxs_iters = [5,10,15,20,50,75,100, 150])

# TODO: stop décommenter

#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.04,0.05, 0.06,0.07], maxs_iters = [400, 500, 600, 750, 1000], verbose = False)


# TODO: stop décommenter

#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_gd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.04,0.05, 0.06,0.07], maxs_iters = [400, 500, 600, 750, 1000], verbose = False)



#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.logistic_regression, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.01,0.05], maxs_iters = [12,20])

# SUBGRADIENT DESCENT:
#print("subgradient descent")
#TODO: run: best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_sgd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [1e-5,1e-4, 0.001, 0.01, 0.02,0.03,0.04, 0.05,0.06,0.07, 0.08, 0.1], maxs_iters = [50,100,200,300,400,500,600,700, 800, 900, 1000, 1100,1200, 1300,1500])
#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_sgd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [1e-5,1e-4,0.001, 0.005,0.01, 0.03], maxs_iters = [25,30,45,50,75,100,150,200,250,375,500])
#loss_train, loss_val = apply_method(im.mean_squared_error_sgd, y_tr, x_tr, y_val = y_val, x_val = x_val, x_te = x_te, id = id, gamma = 0.05, max_iters = 375, validation = True)
#print(loss_train, loss_val)
#best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final = best_triple_param_selection(im.mean_squared_error_sgd, y, x, x_te, id, 10, lambdas = [0.0], gammas = [0.1], maxs_iters = [5,500])

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
