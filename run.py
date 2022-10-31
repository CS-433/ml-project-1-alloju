from methods import (
    best_single_param_selection,
    apply_separation_method,
    best_triple_param_selection,
)
from helpers import load_csv_data
from utilities import load_csv_title
from paths import training_set, test_set
from methods import apply_method, predict
import implementations as im
from preprocessing import preproc_test, preproc_train, to_0_1
import numpy as np

# load the training set
y, x, id_tr = load_csv_data(training_set)
title_tr = load_csv_title(training_set)

# load the test set
_, x_te, id_te = load_csv_data(test_set)
title_te = load_csv_title(test_set)

# create logistic labels for the train set
y_logistic = to_0_1(y)

# reproduce the results posted on AICrowd

apply_separation_method(
    method=im.ridge_regression,
    y_tr=y,
    x_tr=x,
    id_tr=id_tr,
    title_tr=title_tr,
    y_te=_,
    x_te=x_te,
    id_te=id_te,
    title_te=title_te,
    k_fold=10,
    lambdas_=[
        0.0,
        1e-6,
        1e-5,
        2e-4,
        3e-4,
        4e-4,
        5e-4,
        6e-4,
        7e-4,
        8e-4,
        9e-4,
        1e-3,
        1e-2,
        1e-1,
    ],
    initial_w=None,
    max_iters=[1],
    gammas=[0.0],
    do_corr=False,
    do_pca=False,
    percentage=95,
    logistic=False,
    verbose=False,
    do_poly=True,
    degree=6,
)


# # Reproduce the results for each implematation using polynomial feature expansion and one hot encoder

# print("Least squares, degree 5 :")
# apply_separation_method(
#     method=im.least_squares,
#     y_tr=y,
#     x_tr=x,
#     id_tr=id_tr,
#     title_tr=title_tr,
#     y_te=_,
#     x_te=x_te,
#     id_te=id_te,
#     title_te=title_te,
#     k_fold=10,
#     lambdas_=[0.5],
#     initial_w=None,
#     max_iters=[100],
#     gammas=[0.01],
#     do_corr=False,
#     do_pca=False,
#     percentage=95,
#     logistic=False,
#     verbose=True,
#     do_poly=True,
#     degree=5,
# )
# print("Ridge regression, degree 6 :")
# apply_separation_method(
#     method=im.ridge_regression,
#     y_tr=y,
#     x_tr=x,
#     id_tr=id_tr,
#     title_tr=title_tr,
#     y_te=_,
#     x_te=x_te,
#     id_te=id_te,
#     title_te=title_te,
#     k_fold=10,
#     lambdas_=[
#         0.0,
#         1e-6,
#         1e-5,
#         1e-4,
#         2e-4,
#         3e-4,
#         4e-4,
#         5e-4,
#         6e-4,
#         7e-4,
#         8e-4,
#         9e-4,
#         1e-3,
#         1e-2,
#         1e-1,
#     ],
#     initial_w=None,
#     max_iters=[1],
#     gammas=[0.0],
#     do_corr=False,
#     do_pca=False,
#     percentage=95,
#     logistic=False,
#     verbose=True,
#     do_poly=True,
#     degree=6,
# )
# print("Gradient descent, degree 3 :")
# apply_separation_method(
#     method=im.mean_squared_error_gd,
#     y_tr=y,
#     x_tr=x,
#     id_tr=id_tr,
#     title_tr=title_tr,
#     y_te=_,
#     x_te=x_te,
#     id_te=id_te,
#     title_te=title_te,
#     k_fold=10,
#     lambdas_=[0.0],
#     initial_w=None,
#     max_iters=[600],
#     gammas=[3e-4],
#     do_corr=False,
#     do_pca=False,
#     percentage=95,
#     logistic=False,
#     verbose=True,
#     do_poly=True,
#     degree=3,
# )
# print("Stochastic gradient descent, degree 3 :")
# apply_separation_method(
#     method=im.mean_squared_error_sgd,
#     y_tr=y,
#     x_tr=x,
#     id_tr=id_tr,
#     title_tr=title_tr,
#     y_te=_,
#     x_te=x_te,
#     id_te=id_te,
#     title_te=title_te,
#     k_fold=10,
#     lambdas_=[0.0],
#     initial_w=None,
#     max_iters=[20, 30, 70, 100, 300],
#     gammas=[1e-6, 2e-5, 3e-4, 1e-7, 3e-7, 4e-7, 8e-7],
#     do_corr=False,
#     do_pca=False,
#     percentage=95,
#     logistic=False,
#     verbose=False,
#     do_poly=True,
#     degree=3,
# )
# print("Logistic regression, degree 2 :")
# apply_separation_method(
#     method=im.logistic_regression,
#     y_tr=y_logistic,
#     x_tr=x,
#     id_tr=id_tr,
#     title_tr=title_tr,
#     y_te=_,
#     x_te=x_te,
#     id_te=id_te,
#     title_te=title_te,
#     k_fold=10,
#     lambdas_=[0.0],
#     initial_w=None,
#     max_iters=[20, 200, 400],
#     gammas=[1e-6, 3e-5, 1e-4, 3e-4, 4e-3, 1e-3, 1e-2, 2e-1],
#     do_corr=False,
#     do_pca=False,
#     percentage=95,
#     logistic=True,
#     verbose=False,
#     do_poly=True,
#     degree=2,
# )
# print("Regularized logistic regression, degree 3 :")
# apply_separation_method(
#     method=im.reg_logistic_regression,
#     y_tr=y_logistic,
#     x_tr=x,
#     id_tr=id_tr,
#     title_tr=title_tr,
#     y_te=_,
#     x_te=x_te,
#     id_te=id_te,
#     title_te=title_te,
#     k_fold=10,
#     lambdas_=[1e-5],
#     initial_w=None,
#     max_iters=[1000],
#     gammas=[0.1],
#     do_corr=False,
#     do_pca=False,
#     percentage=95,
#     logistic=True,
#     verbose=True,
#     do_poly=True,
#     degree=3,
# )

# # Reproduce the results for the preprocessing on ridge

# print("Polynomial feature expansion, degree 6 and one hot encoding :")
# apply_separation_method(
#     method=im.ridge_regression,
#     y_tr=y,
#     x_tr=x,
#     id_tr=id_tr,
#     title_tr=title_tr,
#     y_te=_,
#     x_te=x_te,
#     id_te=id_te,
#     title_te=title_te,
#     k_fold=10,
#     lambdas_=[
#         0.0,
#         1e-6,
#         1e-5,
#         1e-4,
#         2e-4,
#         3e-4,
#         4e-4,
#         5e-4,
#         6e-4,
#         7e-4,
#         8e-4,
#         9e-4,
#         1e-3,
#         1e-2,
#         1e-1,
#     ],
#     initial_w=None,
#     max_iters=[1],
#     gammas=[0.0],
#     do_corr=False,
#     do_pca=False,
#     percentage=95,
#     logistic=False,
#     verbose=True,
#     do_poly=True,
#     degree=6,
# )

# print("Basic preprocessing :")
# # reload the data
# y, x, id_tr = load_csv_data(training_set)
# title_tr = load_csv_title(training_set)
# _, x_te, id_te = load_csv_data(test_set)
# title_te = load_csv_title(test_set)
# # preprocess the data
# x, x_mean, x_std, ind, projection_matrix = preproc_train(
#     x, title_tr, do_corr=False, do_pca=False, do_poly=False
# )
# x_te = preproc_test(
#     x_te,
#     title_te,
#     x_mean,
#     x_std,
#     projection_matrix,
#     ind,
#     do_corr=False,
#     do_pca=False,
#     do_poly=False,
# )
# # run the method
# (
#     best_param,
#     acc_tr,
#     acc_val,
#     best_loss_tr,
#     best_loss_val,
#     losses_tr,
#     losses_val,
# ) = best_single_param_selection(
#     im.ridge_regression,
#     y,
#     x,
#     x_te,
#     id_te,
#     10,
#     params=[
#         0.0,
#         1e-6,
#         1e-5,
#         1e-4,
#         2e-4,
#         3e-4,
#         5e-4,
#         6e-4,
#         7e-4,
#         8e-4,
#         9e-4,
#         1e-3,
#         1e-2,
#         1e-1,
#     ],
#     tuned_param="lambda",
# )

# print("Basic preprocessing, delete correlated and PCA :")
# # reload the data
# y, x, id_tr = load_csv_data(training_set)
# title_tr = load_csv_title(training_set)
# _, x_te, id_te = load_csv_data(test_set)
# title_te = load_csv_title(test_set)
# # preprocess the data
# x, x_mean, x_std, ind, projection_matrix = preproc_train(
#     x, title_tr, do_corr=True, do_pca=True, do_poly=False
# )
# x_te = preproc_test(
#     x_te,
#     title_te,
#     x_mean,
#     x_std,
#     projection_matrix,
#     ind,
#     do_corr=True,
#     do_pca=True,
#     do_poly=False,
# )
# # run the method
# (
#     best_param,
#     acc_tr,
#     acc_val,
#     best_loss_tr,
#     best_loss_val,
#     losses_tr,
#     losses_val,
# ) = best_single_param_selection(
#     im.ridge_regression,
#     y,
#     x,
#     x_te,
#     id_te,
#     10,
#     params=[
#         0.0,
#         1e-6,
#         1e-5,
#         1e-4,
#         2e-4,
#         3e-4,
#         5e-4,
#         6e-4,
#         7e-4,
#         8e-4,
#         9e-4,
#         1e-3,
#         1e-2,
#         1e-1,
#     ],
#     tuned_param="lambda",
# )

# print("Polynomial feature expansion on ridge regression, degree 8 :")
# # reload the data
# y, x, id_tr = load_csv_data(training_set)
# title_tr = load_csv_title(training_set)
# _, x_te, id_te = load_csv_data(test_set)
# title_te = load_csv_title(test_set)
# # preprocess the data
# x, x_mean, x_std, ind, projection_matrix = preproc_train(
#     x, title_tr, do_corr=False, do_pca=False, do_poly=True, degree=8
# )
# x_te = preproc_test(
#     x_te,
#     title_te,
#     x_mean,
#     x_std,
#     projection_matrix,
#     ind,
#     do_corr=False,
#     do_pca=False,
#     do_poly=True,
#     degree=8,
# )
# # run the method
# (
#     best_param,
#     acc_tr,
#     acc_val,
#     best_loss_tr,
#     best_loss_val,
#     losses_tr,
#     losses_val,
# ) = best_single_param_selection(
#     im.ridge_regression,
#     y,
#     x,
#     x_te,
#     id_te,
#     10,
#     params=[
#         0.0,
#         1e-6,
#         1e-5,
#         1e-4,
#         2e-4,
#         3e-4,
#         5e-4,
#         6e-4,
#         7e-4,
#         8e-4,
#         9e-4,
#         1e-3,
#         1e-2,
#         1e-1,
#     ],
#     tuned_param="lambda",
# )

# # Example of cross validation:

# print(
#     "Cross Validation on regularized logistic regression, using polynomial expansion of degree 3:"
# )
# y, x, id_tr = load_csv_data(training_set)
# title_tr = load_csv_title(training_set)
# _, x_te, id_te = load_csv_data(test_set)
# title_te = load_csv_title(test_set)
# # preprocess the data
# x, x_mean, x_std, ind, projection_matrix = preproc_train(
#     x, title_tr, do_corr=False, do_pca=False, do_poly=True, degree=3
# )
# x_te = preproc_test(
#     x_te,
#     title_te,
#     x_mean,
#     x_std,
#     projection_matrix,
#     ind,
#     do_corr=False,
#     do_pca=False,
#     do_poly=True,
#     degree=3,
# )

# # Example of cross validation on regularized logistic regression :
# (
#     best_lambda,
#     best_gamma,
#     best_max_iters,
#     accuracy_tr,
#     accuracy_val,
# ) = best_triple_param_selection(
#     im.reg_logistic_regression,
#     y_logistic,
#     x,
#     x_te,
#     id,
#     10,
#     lambdas=[1e-5, 1e-4],
#     maxs_iters=[500, 1000],
#     gammas=[0.1, 0.5],
#     logistic=True,
# )
