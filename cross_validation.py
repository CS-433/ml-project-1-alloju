import numpy as np
from apply_method import apply_method, predict
from utilities import compute_mse, compute_rmse
from plots import cross_validation_visualization

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

def cross_validation(method, y, x, k_indices, k, lambda_ = 0.5, initial_w = None, max_iters = 100, gamma = 0.1):
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

    loss_tr, loss_val = apply_method(method, y_tr, x_tr, y_val = y_val, x_val = x_val, lambda_ = lambda_, cross_val= True)

    return loss_tr, loss_val

def best_single_param_selection(method, y,x, x_te, id, k_fold, params = [0.1, 0.5], tuned_param = "", lambda_ = 0.1, initial_w = None, max_iters = 10, gamma = 0.1, seed = 1):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        params: shape = (p, ) where p is the number of values of tuned parameter to test. 
        tuned_param: name of parameters to tune. Possibilities are: lambda, initial_w, max_iters, gamma
        lambda_: value if lambda is not tuned
        initial_ws: initial weights
        maxs_iters: nb maximal of iterations
        gamma: value if gamma is not tuned
        seed: fixed seed
        
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_val = []
    # cross validation over lambdas
    for param in params:
        temp_rmse_tr = []
        temp_rmse_val = []
        for k in range(k_fold):
            if tuned_param == "lambda":
                loss_tr, loss_val= cross_validation(method, y , x, k_indices, k, lambda_ = param, initial_w = None, max_iters = 10, gamma = 0.1)
            elif tuned_param == "gamma":
                loss_tr, loss_val= cross_validation(method, y , x, k_indices, k, gamma = param, lambda_ = 0.1, initial_w = None, max_iters = 10)
            elif tuned_param == "max_iters":
                loss_tr, loss_val= cross_validation(method, y , x, k_indices, k, max_iters = param, lambda_ = 0.1, initial_w = None, gamma = 0.1)
            else:
                print("Please specify which parameter you are tuning")
                return 0
            temp_rmse_tr.append(loss_tr)
            temp_rmse_val.append(loss_val)
        rmse_tr.append(np.mean(temp_rmse_tr))
        rmse_val.append(np.mean(temp_rmse_val))
        print("tuned_param = ", param, "rmse_tr = ", np.mean(temp_rmse_tr), "rmse_val", np.mean(temp_rmse_val))
    best_rmse_val = (min(rmse_val))
    idx = np.where(rmse_val == best_rmse_val)
    best_rmse_tr = rmse_tr[np.squeeze(idx)]
    best_param = params[np.squeeze(idx)]

    cross_validation_visualization(params, rmse_tr, rmse_val)

    if tuned_param == "lambda":
        rmse_tr_final = apply_method(method, y, x, x_te = x_te, id = id, lambda_ = best_param, initial_w = None, max_iters = 10, gamma = 0.1, validation = False)
    elif tuned_param == "gamma":
        rmse_tr_final = apply_method(method, y, x, x_te = x_te, id = id, gamma = best_param, lambda_ = 0.1, initial_w = None, max_iters = 10, validation = False)
    elif tuned_param == "max_iters":
        rmse_tr_final = apply_method(method, y, x, x_te = x_te, id = id, max_iters = best_param, lambda_ = 0.1, initial_w = None, gamma = 0.1, validation = False)

    #rmse_tr_final, _ = apply_method(method, y, x, np.zeros_like(y), np.zeros_like(x), x_te, id, best_param, validation = False)
    print("final training rmse", rmse_tr_final)
    print("Chosen" + tuned_param + "is: ", best_param)

    return best_param, best_rmse_val, best_rmse_tr

def best_triple_param_selection(method, y,x, x_te, id, k_fold, lambdas = [0.1, 0.5], gammas =[0.1,0.5], maxs_iters = [5,10], initial_w = None, seed = 1):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        params: shape = (p, ) where p is the number of values of tuned parameter to test. 
        tuned_param: name of parameters to tune. Possibilities are: lambda, initial_w, max_iters, gamma
        lambda_: value if lambda is not tuned
        initial_ws: initial weights
        maxs_iters: nb maximal of iterations
        gamma: value if gamma is not tuned
        seed: fixed seed
        
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    super_best_rmse_val = []
    best_params1 = []
    best_gammas = []
    super_best_iter = []
    # cross validation over lambdas
    for lambda_ in lambdas:
        best_rmse_val = []
        best_max_iters = []
        for gamma in gammas:
            rmse_val = []
            for max_iters in maxs_iters:
                temp_rmse_val = []
                for k in range(k_fold):
                    loss_tr, loss_val= cross_validation(method, y , x, k_indices, k, lambda_ = lambda_, initial_w = None, max_iters = max_iters, gamma = gamma)
                    temp_rmse_val.append(loss_val)
                rmse_val.append(np.mean(temp_rmse_val))
            
            best_temp_rmse = min(rmse_val)
            best_rmse_val.append(best_temp_rmse)
            best_max_iters.append(maxs_iters[np.argmin(rmse_val)])
        best_rmse = min(best_rmse_val)
        best_max_iter = best_max_iters[np.argmin(best_rmse_val)]
        super_best_iter.append(best_max_iter)
        best_gammas.append(gammas[np.argmin(best_rmse_val)])
        super_best_rmse_val.append(best_rmse)
    super_best_rmse = min(super_best_rmse_val)
    idx_super_best = np.argmin(super_best_rmse_val)
    best_lambda = lambdas[idx_super_best]
    best_gamma = best_gammas[idx_super_best]
    best_max_iters = super_best_iter[idx_super_best]
        
    print("lambda = ", best_lambda, "max_iters = ", best_max_iters, "gamma = ", best_gamma, "rmse_tr = ", super_best_rmse)

    #cross_validation_visualization(params, rmse_tr, rmse_val)

    rmse_tr_final = apply_method(method, y, x, x_te = x_te, id = id, max_iters = best_max_iter, lambda_ = best_lambda, initial_w = None, gamma = best_gamma, validation = False)

    #rmse_tr_final, _ = apply_method(method, y, x, np.zeros_like(y), np.zeros_like(x), x_te, id, best_param, validation = False)
    print("final training rmse", rmse_tr_final)

    return best_lambda, best_gamma, best_max_iters, best_rmse_val, rmse_tr_final


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