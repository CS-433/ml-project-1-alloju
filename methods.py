import numpy as np
import os.path as op
from plots import (
    cross_validation_visualization,
    cross_validation_visualization_degree,
    cross_validation_visualization_multiple,
)
from helpers import load_csv_data, create_csv_submission
from preprocessing import (
    preproc_test,
    preproc_train,
    to_0_1,
    replace_class,
    class_separation,
)
from paths import prediction_dir, training_set, test_set
from utilities import (
    compute_accuracy,
    compute_loss_neg_loglikelihood,
    compute_mse,
    sigmoid,
    split_data,
    load_csv_title,
)


def apply_method(
    method,
    y_tr,
    x_tr,
    y_val=np.zeros([10, 1]),
    x_val=np.zeros([10, 1]),
    x_te=np.zeros([5, 1]),
    id=np.zeros(5),
    lambda_=0.5,
    initial_w=None,
    max_iters=100,
    gamma=0.01,
    do_predictions=True,
    validation=True,
    loss="original",
    logistic=False,
    separation=False,
):
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
    if initial_w == None:
        initial_w = np.zeros(x_tr.shape[1])
    loss_tr = 0
    loss_val = 0  # To avoid problem if no validation
    if "reg_logistic_regression" in str(method):
        w, neg_log_likelihood_tr = method(
            y_tr, x_tr, lambda_, initial_w, max_iters, gamma
        )
        loss_tr = neg_log_likelihood_tr
    elif "logistic_regression" in str(method):
        w, neg_log_likelihood_tr = method(y_tr, x_tr, initial_w, max_iters, gamma)
        loss_tr = neg_log_likelihood_tr
    elif "mean_squared_error" in str(method) or "mean_squared_error_sgd" in str(method):
        w, mse = method(y_tr, x_tr, initial_w, max_iters, gamma)
        loss_tr = mse
    elif "least_squares" in str(method):
        w, mse = method(y_tr, x_tr)
        loss_tr = mse
    elif "ridge_regression" in str(method):
        w, mse = method(y_tr, x_tr, lambda_)
        loss_tr = mse

    # avoid error if no validation set
    loss_val = 0

    if loss == "original":
        if "logistic_regression" in str(method):
            loss_tr = neg_log_likelihood_tr
            if validation:
                loss_val = compute_loss_neg_loglikelihood(y_val, x_val, w)
        else:
            loss_tr = mse
            if validation:
                loss_val = compute_mse(y_val, x_val, w)
    elif loss == "accuracy":
        loss_tr = compute_accuracy(y_tr, x_tr, w, logistic)
        loss_val = compute_accuracy(y_val, x_val, w, logistic)

    if do_predictions:
        if separation:
            y_bin = predict(method, id, x_te, w, separation)
            return loss_tr, loss_val, y_bin
        predict(method, id, x_te, w)

    return loss_tr, loss_val


def predict(method, id, x_te, w, separation=False):
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
    y = np.dot(x_te, w)
    # Transform the labels in the right format for the submission
    y_bin = sigmoid(y)
    y_bin[y_bin < 0.5] = -1
    y_bin[y_bin >= 0.5] = 1
    if separation:
        return y_bin  # Only a fraction of the prediction
    path = op.join(prediction_dir, "prediction" + str(method) + ".csv")
    create_csv_submission(id, y_bin, path)


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(
    method,
    y,
    x,
    k_indices,
    k,
    lambda_=0.5,
    initial_w=None,
    max_iters=100,
    gamma=0.1,
    logistic=False,
):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        method:     method to apply to the data
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, regularisation parameter
        initial_w:  the initial weight
        max_iters:  the number of iteration maximal
        gamma:      the learning rate
        logistic:   Boolean; indicates if we have a logistic method

    Returns:
        loss_tr:    loss of the training set
        loss_val:   loss of the validation
    """
    # get k'th subgroup in validation, others in train:
    valid_indices = k_indices[k]
    train_indices = np.delete(k_indices, k, axis=0).reshape(-1)
    x_tr = x[train_indices]
    x_val = x[valid_indices]
    y_tr = y[train_indices]
    y_val = y[valid_indices]

    loss_tr, loss_val = apply_method(
        method,
        y_tr,
        x_tr,
        y_val=y_val,
        x_val=x_val,
        lambda_=lambda_,
        initial_w=initial_w,
        max_iters=max_iters,
        gamma=gamma,
        do_predictions=False,
        logistic=logistic,
    )

    return loss_tr, loss_val


def best_single_param_selection(
    method,
    y,
    x,
    x_te,
    id,
    k_fold,
    params=[0.1, 0.5],
    tuned_param="",
    lambda_=0.1,
    initial_w=None,
    max_iters=10,
    gamma=0.1,
    seed=1,
    verbose=True,
    logistic=False,
):
    """cross validation over regularisation parameter lambda.

    Args:
        method:         method to apply to the data
        y:              labels
        x:              features
        x_te:           test features
        id:             index of the labels
        k_fold:         integer, the number of folds
        params:         shape = (p, ) where p is the number of values of tuned parameter to test
        tuned_param:    name of parameters to tune. Possibilities are: lambda, max_iters, gamma
        lambda_:        regularisation parameter
        initial_ws:     initial weights
        max_iters:      the number maximal of iterations
        gamma:          the learning rate
        seed:           fixed seed
        verbose:        boolean; specifies if the intermediate losses and accuracies are printed
        logistic:       boolean; indicates if we have a logistic method

    Returns:
        best_param:     best value of the tune parameter
        acc_tr:         accuracy of the training set
        acc_val:        accuracy of the validation set
        best_loss_tr:   best loss of the training set
        best_loss_val:  best loss of the validsation set
        losses_tr:      losses of the training set
        losses_val:     losses of the validation set
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and validation data
    losses_tr = []
    losses_val = []
    # cross validation over tuned parameter
    for param in params:
        temp_loss_tr = []
        temp_loss_val = []
        for k in range(k_fold):
            if tuned_param == "lambda":
                loss_tr, loss_val = cross_validation(
                    method,
                    y,
                    x,
                    k_indices,
                    k,
                    lambda_=param,
                    initial_w=initial_w,
                    max_iters=max_iters,
                    gamma=gamma,
                    logistic=logistic,
                )
            elif tuned_param == "gamma":
                loss_tr, loss_val = cross_validation(
                    method,
                    y,
                    x,
                    k_indices,
                    k,
                    gamma=param,
                    lambda_=lambda_,
                    initial_w=initial_w,
                    max_iters=max_iters,
                    logistic=logistic,
                )
            elif tuned_param == "max_iters":
                loss_tr, loss_val = cross_validation(
                    method,
                    y,
                    x,
                    k_indices,
                    k,
                    max_iters=param,
                    lambda_=lambda_,
                    initial_w=initial_w,
                    gamma=gamma,
                    logistic=logistic,
                )
            else:
                raise ValueError("Please specify which parameter you are tuning")
            if np.isnan(loss_val):
                # to avoid that the cross val takes nan as the min !
                loss_tr = 10000
                loss_val = 10000

            temp_loss_tr.append(loss_tr)
            temp_loss_val.append(loss_val)
        losses_tr.append(np.mean(temp_loss_tr))
        losses_val.append(np.mean(temp_loss_val))
        if verbose:
            print(
                "tuned_param = ",
                param,
                "loss_tr = ",
                np.mean(temp_loss_tr),
                "loss_val",
                np.mean(temp_loss_val),
            )
    best_loss_val = min(losses_val)
    idx = np.where(losses_val == best_loss_val)
    best_loss_tr = losses_tr[np.squeeze(idx)]
    best_param = params[np.squeeze(idx)]

    cross_validation_visualization(
        method, params, losses_tr, losses_val, tuned_param, 0
    )

    x_tr, x_val, y_tr, y_val = split_data(x, y, 0.8)

    if tuned_param == "lambda":
        loss_tr_final, _ = apply_method(
            method,
            y,
            x,
            x_te=x_te,
            id=id,
            lambda_=best_param,
            initial_w=initial_w,
            max_iters=max_iters,
            gamma=gamma,
            validation=False,
            logistic=logistic,
        )
        acc_tr, acc_val = apply_method(
            method,
            y_tr,
            x_tr,
            y_val,
            x_val,
            lambda_=best_param,
            initial_w=initial_w,
            max_iters=max_iters,
            gamma=gamma,
            validation=True,
            loss="accuracy",
            do_predictions=False,
            logistic=logistic,
        )
    elif tuned_param == "gamma":
        loss_tr_final, _ = apply_method(
            method,
            y,
            x,
            x_te=x_te,
            id=id,
            gamma=best_param,
            lambda_=lambda_,
            initial_w=initial_w,
            max_iters=max_iters,
            validation=False,
            logistic=logistic,
        )
        acc_tr, acc_val = apply_method(
            method,
            y_tr,
            x_tr,
            y_val,
            x_val,
            gamma=best_param,
            lambda_=lambda_,
            initial_w=initial_w,
            max_iters=max_iters,
            validation=True,
            loss="accuracy",
            do_predictions=False,
            logistic=logistic,
        )

    elif tuned_param == "max_iters":
        loss_tr_final, _ = apply_method(
            method,
            y,
            x,
            x_te=x_te,
            id=id,
            max_iters=best_param,
            lambda_=lambda_,
            initial_w=initial_w,
            gamma=gamma,
            validation=False,
            logistic=logistic,
        )
        acc_tr, acc_val = apply_method(
            method,
            y_tr,
            x_tr,
            y_val,
            x_val,
            max_iters=best_param,
            lambda_=lambda_,
            initial_w=initial_w,
            gamma=gamma,
            validation=True,
            loss="accuracy",
            do_predictions=False,
            logistic=logistic,
        )

    print("accuracy measures: ", "train = ", acc_tr, "val = ", acc_val)
    print("final training loss", loss_tr_final)
    print("Chosen " + tuned_param + " is: ", best_param)

    return (
        best_param,
        acc_tr,
        acc_val,
        best_loss_tr,
        best_loss_val,
        losses_tr,
        losses_val,
    )


def best_triple_param_selection(
    method,
    y,
    x,
    x_te,
    id,
    k_fold,
    lambdas=[0.1, 0.5],
    gammas=[0.1, 0.5],
    maxs_iters=[5, 10],
    initial_w=None,
    seed=1,
    verbose=True,
    separation=False,
    logistic=False,
):
    """cross validation over regularisation parameter lambda.

    Args:
        method:         method to apply to the data
        y:              labels
        x:              features
        x_te:           test features
        id:             index of the labels
        k_fold:         integer, the number of folds
        lambdas:        regularisation parameter
        gammas:          the learning rate
        max_iters:      the number maximal of iterations
        initial_ws:     initial weights
        seed:           fixed seed
        verbose:        boolean; specifies if the intermediate losses and accuracies are printed
        separation:     boolean; indicates if we use the function for the whole dataset or a fraction of the dataset
        logistic:       boolean; indicates if we have a logistic method

    Returns:
        best_lambda:    scalar, value of the best lambda
        best_gamma:     scalar, value of the best gamma
        best_max_iters: scalar, value of the best max_iters
        acc_tr:         if separation = False:
        acc_val:        if separation = False:
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training and validation datas, the best gammas and best lambdas
    super_best_loss_val = []
    super_best_loss_train = []
    best_gammas = []
    super_best_lambdas = []

    for max_iters in maxs_iters:
        best_loss_val = []
        best_loss_train = []
        best_lambdas = []

        for gamma in gammas:
            losses_val = []
            losses_train = []

            for lambda_ in lambdas:
                temp_loss_val = []
                temp_loss_train = []

                for k in range(k_fold):
                    loss_tr, loss_val = cross_validation(
                        method,
                        y,
                        x,
                        k_indices,
                        k,
                        lambda_=lambda_,
                        initial_w=initial_w,
                        max_iters=max_iters,
                        gamma=gamma,
                        logistic=logistic,
                    )
                    if np.isnan(loss_val):
                        # to avoid that the cross val takes nan as the min !
                        loss_tr = 10000
                        loss_val = 10000

                    temp_loss_val.append(loss_val)
                    temp_loss_train.append(loss_tr)
                losses_val.append(np.mean(temp_loss_val))
                losses_train.append(np.mean(temp_loss_train))
                if verbose:
                    print(
                        "For: lambda = ",
                        lambda_,
                        " gamma = ",
                        gamma,
                        " max_iters = ",
                        max_iters,
                        ", training loss = ",
                        np.mean(temp_loss_train),
                        " validation loss = ",
                        np.mean(temp_loss_val),
                    )

            best_temp_loss = min(losses_val)
            best_loss_val.append(best_temp_loss)
            best_loss_train.append(min(losses_train))
            best_lambdas.append(lambdas[np.argmin(losses_val)])

        best_loss = min(best_loss_val)
        best_lambda = best_lambdas[np.argmin(best_loss_val)]
        super_best_lambdas.append(best_lambda)
        best_gammas.append(gammas[np.argmin(best_loss_val)])
        super_best_loss_val.append(best_loss)
        super_best_loss_train.append(min(best_loss_train))

    super_best_loss = min(super_best_loss_val)
    idx_super_best = np.argmin(super_best_loss_val)
    best_max_iters = maxs_iters[idx_super_best]
    best_gamma = best_gammas[idx_super_best]
    super_best_lambda = super_best_lambdas[idx_super_best]

    print(
        "Chosen parameters are: ",
        "lambda = ",
        super_best_lambda,
        "max_iters = ",
        best_max_iters,
        "gamma = ",
        best_gamma,
        "loss train = ",
        min(super_best_loss_train),
        "loss val = ",
        super_best_loss,
    )
    if separation:
        return best_lambda, best_gamma, best_max_iters
    # cross_validation_visualization(params, loss_tr, loss_val)
    x_tr, x_val, y_tr, y_val = split_data(x, y, 0.8)
    acc_tr, acc_val = apply_method(
        method,
        y_tr,
        x_tr,
        y_val,
        x_val,
        max_iters=best_max_iters,
        lambda_=best_lambda,
        initial_w=initial_w,
        gamma=best_gamma,
        validation=True,
        loss="accuracy",
        do_predictions=False,
        logistic=logistic,
    )
    print("accuracy measures: ", "train = ", acc_tr, "val = ", acc_val)
    loss_tr_final, _ = apply_method(
        method,
        y,
        x,
        x_te=x_te,
        id=id,
        max_iters=best_max_iters,
        lambda_=best_lambda,
        initial_w=initial_w,
        gamma=best_gamma,
        validation=False,
        logistic=logistic,
    )
    print("final training loss", loss_tr_final)

    return best_lambda, best_gamma, best_max_iters, acc_tr, acc_val


def best_degree_selection(
    method,
    k_fold,
    degrees=[2, 4],
    params=[0.1, 0.5],
    tuned_param="",
    lambda_=0.1,
    initial_w=None,
    max_iters=10,
    gamma=0.1,
    seed=1,
    verbose=True,
    logistic=False,
):
    """Allow to select the best degree, based on a cross validation result of a method. The best degree isn't cross-validated to save running time.
    Also, cross validation on only one parameter is allowed, in order not to have multiple

    Args:
        method:     method to apply to the data
        k_fold:     integer, the number of folds
        degrees:    degrees to select for the best polynomial
        params:     possible value for the tuned parameter
        tuned_param:name of parameters to tune. Possibilities are: lambda, max_iters, gamma
        lambda_:    regularisation parameter
        initial_w:  initial weights
        max_iters:  the number maximal of iterations
        gamma:      the learning rate
        seed:       fixed seed
        verbose:    boolean; specifies if the intermediate losses and accuracies are printed
        logistic:   boolean; indicates if we have a logistic method

    Returns:
        best_param:         the best tuned parameter value
        best_degree:        the best degree value
        best_accuracy_tr:   the best accuracy of the training data
        best_accuracy_val:  the best accuracy of the validation data
    """
    cross_mse_tr = []
    cross_mse_val = []
    tuned_param = "lambda"
    for chosen_degree in degrees:
        y, x, ids = load_csv_data(training_set)
        title = load_csv_title(training_set)

        print("degree= ", chosen_degree)
        x, x_mean, x_std, ind, projection_matrix = preproc_train(
            x, title, do_corr=False, do_pca=False, do_poly=True, degree=chosen_degree
        )

        _, x_te, id = load_csv_data(test_set)
        title = load_csv_title(test_set)

        x_te = preproc_test(
            x_te,
            title,
            x_mean,
            x_std,
            projection_matrix,
            ind,
            do_corr=False,
            do_pca=False,
            do_poly=True,
            degree=chosen_degree,
        )

        if verbose:
            print("Data have been preprocessed")

        if logistic:
            y_logistic = to_0_1(y)
            (
                best_param,
                cross_mse_tr,
                cross_mse_val,
                best_loss_tr,
                best_loss_val,
                losses_tr,
                losses_val,
            ) = best_single_param_selection(
                method,
                y_logistic,
                x,
                x_te,
                id,
                k_fold,
                params=params,
                tuned_param=tuned_param,
                logistic=True,
                verbose=verbose,
            )
        else:
            (
                best_param,
                cross_mse_tr_rr,
                cross_mse_val_rr,
                best_loss_tr,
                best_loss_val,
                losses_tr,
                losses_val,
            ) = best_single_param_selection(
                method,
                y,
                x,
                x_te,
                id,
                k_fold,
                params=params,
                tuned_param=tuned_param,
                logistic=False,
                verbose=verbose,
            )
        cross_mse_tr.append(cross_mse_tr_rr)
        cross_mse_val.append(cross_mse_val_rr)
        cross_validation_visualization_multiple(
            str(method)[1:-19],
            params,
            losses_tr,
            losses_val,
            tuned_param,
            chosen_degree,
            1,
        )

    best_accuracy_val = max(cross_mse_val)
    idx = np.where(cross_mse_val == best_accuracy_val)
    best_accuracy_tr = cross_mse_tr[np.squeeze(idx)]
    best_degree = degrees[np.squeeze(idx)]

    print(
        "Chosen degree = ",
        best_degree,
        " accuracies: training: ",
        best_accuracy_tr,
        ", validation: ",
        best_accuracy_val,
    )

    cross_validation_visualization_degree(
        str(method)[1:-19] + "_degree_selection",
        degrees,
        cross_mse_tr,
        cross_mse_val,
        "degree",
        2,
    )
    return best_param, best_degree, best_accuracy_tr, best_accuracy_val


def joining_prediction(method, id, y):
    """Create csv submission of the prediction from the different classes
    Args:
        method: method to apply to the data
        id:     index of the y labels
        y:      the labels
    """
    path = op.join(prediction_dir, "prediction" + str(method) + ".csv")
    create_csv_submission(id, y, path)


def apply_separation_method(
    method,
    y_tr,
    x_tr,
    id_tr,
    title_tr,
    y_te,
    x_te,
    id_te,
    title_te,
    k_fold=10,
    lambdas_=[0.5],
    initial_w=None,
    max_iters=[100],
    gammas=[0.01],
    do_corr=False,
    do_pca=False,
    percentage=95,
    logistic=False,
    verbose=True,
    do_poly=False,
    degree=0,
):
    """
    Args:
        verbose = True, do_poly = False, degree = 0
        method:     method to apply to the data
        y_tr:       training labels
        x_tr:       training features
        id_tr:      index of the training labels
        title_tr:   name of the training features
        y_te:       test labels
        x_te:       test features
        id_te:      ndex of the test labels
        title_te:   name of the test features
        k_fold:     integer, the number of folds
        lambdas_:   regularisation parameter
        initial_w:  the initial weight
        max_iters:  the number of iteration maximal
        gammas:     the learning rate
        do_corr:    boolean; indicates if the deletion of correlated features is applied
        do_pca:     boolean; indicates if the PCA is applied
        percentage: percentage for the PCA
        logistic:   boolean; indicates if we have a logistic method
        verbose:    boolean; specifies if the intermediate losses and accuracies are printed
        do_poly:    boolean; indicates if the polynomial feature expansion is performed
        degree:     degree for the polynomial feature expansion
    """
    # Separation of the training and test data according to the classes
    x_tr, title_tr = replace_class(x_tr, title_tr)
    xs_tr, ys_tr, ids_ = class_separation(x_tr, title_tr, id_tr, y_tr)
    x_te, title_te = replace_class(x_te, title_te)
    xs_te, _, ids_te = class_separation(x_te, title_te, id_te, y_te)

    # Definition of lists to store values
    mse_trains = []
    ys = []
    ids = []
    acc_trains = []
    acc_vals = []

    for i in range(len(xs_tr)):
        xi_tr = xs_tr[i]
        yi_tr = ys_tr[i]
        xi_te = xs_te[i]
        idi_te = ids_te[i]

        print("For class ", i, ":")

        # Preprocessing
        xi_tr, x_mean, x_std, ind, projection_matrix = preproc_train(
            xi_tr,
            title_tr,
            percentage=percentage,
            do_corr=do_corr,
            do_pca=do_pca,
            do_poly=do_poly,
            degree=degree,
        )  #  do_poly = do_poly
        xi_te = preproc_test(
            xi_te,
            title_te,
            x_mean,
            x_std,
            projection_matrix,
            ind,
            do_corr=do_corr,
            do_pca=do_pca,
            do_poly=do_poly,
            degree=degree,
        )  # do_poly = do_poly

        # Tune for the best param
        best_lambda, best_gamma, best_max_iters = best_triple_param_selection(
            method,
            y=yi_tr,
            x=xi_tr,
            x_te=xi_te,
            id=idi_te,
            k_fold=k_fold,
            lambdas=lambdas_,
            gammas=gammas,
            maxs_iters=max_iters,
            initial_w=initial_w,
            seed=1,
            verbose=verbose,
            separation=True,
        )

        # prepare predictions
        mse_train, _, y_bin = apply_method(
            method,
            yi_tr,
            xi_tr,
            x_te=xi_te,
            id=idi_te,
            lambda_=best_lambda,
            initial_w=initial_w,
            max_iters=best_max_iters,
            gamma=best_gamma,
            loss="original",
            validation=False,
            separation=True,
            do_predictions=True,
            logistic=logistic,
        )

        # calculate accuracies
        x_tr_sep, x_val_sep, y_tr_sep, y_val_sep = split_data(xi_tr, yi_tr, 0.8)
        acc_train, acc_val = apply_method(
            method,
            y_tr_sep,
            x_tr_sep,
            y_val=y_val_sep,
            x_val=x_val_sep,
            lambda_=best_lambda,
            initial_w=initial_w,
            max_iters=best_max_iters,
            gamma=best_gamma,
            do_predictions=False,
            validation=True,
            loss="accuracy",
            logistic=logistic,
            separation=False,
        )
        print("Accuracy:", acc_train, acc_val)

        ratio = len(xi_tr) / len(x_tr)
        mse_trains.append(mse_train * ratio)
        acc_trains.append(acc_train * ratio)
        acc_vals.append(acc_val * ratio)
        ys = np.concatenate((ys, y_bin))
        ids = np.concatenate((ids, idi_te))

    ids = np.squeeze(ids)
    ids_ys = np.array([ids.T, ys.T], np.int32)
    # Sort in the same order than in the load_csv_data input
    index = np.argsort(ids_ys[0])
    ids_ys[1] = ids_ys[1][index]
    ids_ys[0] = ids_ys[0][index]

    joining_prediction(method, ids_ys[0], ids_ys[1])
    print(
        "accuracy measures: ",
        "train = ",
        np.sum(acc_trains),
        "val = ",
        np.sum(acc_vals),
    )
    print("final training loss", np.sum(mse_trains))
