import numpy as np
from apply_method import apply_method
from utilities import compute_rmse

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
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(method, y, x, x_te, k_indices, k, lambda_ = 0.5, initial_w = np.array([0]), max_iters = 100, gamma = 0.1):
    #TODO: si on a pas de initial w c'est qu'on l'utilise pas non? Donc on peut y mettre n'importe quoi ?
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
    """

    # get k'th subgroup in validation, others in train: 
    valid_indices = k_indices[k]
    train_indices = np.delete(k_indices,k, axis = 0).reshape(-1)
    x_tr = x[train_indices]
    x_val = x[valid_indices]
    y_tr = y[train_indices]
    y_val = y[valid_indices]  

    w_tr = apply_method(method, y_tr, x_tr,y_val,x_val, x_te, lambda_)

    loss_tr = compute_rmse(y_tr, x_tr, w_tr)
    loss_te = compute_rmse(y_val, x_val, w_tr)

    return loss_tr, loss_te, w_tr

def best_lambda_selection(method, y,x, x_te, k_fold, lambdas = [0.1, 0.5], initial_ws = np.array([0]), maxs_iters = 10, gammas = 0.1, seed = 1):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation over lambdas
    for lambda_ in lambdas:
        temp_rmse_tr = []
        temp_rmse_te = []
        for k in range(k_fold):
            loss_tr, loss_te, w_tr = cross_validation(method, y , x, x_te, k_indices, k, lambda_)
            temp_rmse_tr.append(loss_tr)
            temp_rmse_te.append(loss_te)
        rmse_tr.append(np.mean(temp_rmse_tr))
        rmse_te.append(np.mean(temp_rmse_te))
    best_rmse = (min(rmse_te))
    best_lambda = lambdas[np.where(rmse_te == best_rmse)]

    return best_lambda, best_rmse

def best_lambda_and_maxiters_selection(method, y, x, x_te, max_iters, k_fold, lambdas, seed = 1):
    """cross validation over regularisation parameter lambda and degree.
    
    Args:
        degrees: shape = (d,), where d is the number of degrees to test 
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)
        
    """
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation over degrees and lambdas: TODO
    #rmse_tr = [] isn't useful
    best_rmses_te = []
    best_lambdas = []
    for max_iter in max_iters: 
        rmse_te = []
        for lambda_ in lambdas:
            #temp_rmse_tr = []
            temp_rmse_te = []
            for k in range(k_fold):
                loss_tr, loss_te, w_tr = cross_validation(method, y, x, x_te, k_indices, k, lambda_, max_iter)
                #temp_rmse_tr.append(loss_tr)
                temp_rmse_te.append(loss_te)
            #temp_lambda_rmse_tr.append(np.mean(temp_rmse_tr))
            rmse_te.append(np.mean(temp_rmse_te))
            
        best_temp_rmse = min(rmse_te)
        best_rmses_te.append(best_temp_rmse) 
        best_lambdas.append(lambdas[np.argmin(temp_rmse_te)])
        #rmse_tr.append(np.mean(temp_lambda_rmse_tr))
    best_rmse = (min(best_rmses_te))
    best_lambda = best_lambdas[np.argmin(best_rmses_te)]
    best_max_iter = max_iters[np.argmin(best_rmses_te)]
    # ***************************************************
    #raise NotImplementedError    
    
    return best_max_iter, best_lambda, best_rmse