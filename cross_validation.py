import numpy as np
from apply_method import apply_method, predict
from utilities import compute_mse, compute_mse
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
        train and test root mean square errors mse = sqrt(2 mse)
    """

    # get k'th subgroup in validation, others in train: 
    valid_indices = k_indices[k]
    train_indices = np.delete(k_indices,k, axis = 0).reshape(-1)
    x_tr = x[train_indices]
    x_val = x[valid_indices]
    y_tr = y[train_indices]
    y_val = y[valid_indices]  

    loss_tr, loss_val = apply_method(method, y_tr, x_tr, y_val = y_val, x_val = x_val, lambda_ = lambda_, initial_w = initial_w, max_iters = max_iters, gamma = gamma, cross_val= True)

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
        best_mse : scalar, the associated root mean squared error for the best lambda
    """
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mse_tr = []
    mse_val = []
    # cross validation over tuned parameter
    for param in params:
        temp_mse_tr = []
        temp_mse_val = []
        for k in range(k_fold):
            if tuned_param == "lambda":
                loss_tr, loss_val= cross_validation(method, y , x, k_indices, k, lambda_ = param, initial_w = initial_w, max_iters = max_iters, gamma = gamma)
            elif tuned_param == "gamma":
                loss_tr, loss_val= cross_validation(method, y , x, k_indices, k, gamma = param, lambda_ = lambda_, initial_w = initial_w, max_iters = max_iters)
            elif tuned_param == "max_iters":
                loss_tr, loss_val= cross_validation(method, y , x, k_indices, k, max_iters = param, lambda_ = lambda_, initial_w = initial_w, gamma = gamma)
            else:
                print("Please specify which parameter you are tuning")
                return 0
            temp_mse_tr.append(loss_tr)
            temp_mse_val.append(loss_val)
        mse_tr.append(np.mean(temp_mse_tr))
        mse_val.append(np.mean(temp_mse_val))
        print("tuned_param = ", param, "mse_tr = ", np.mean(temp_mse_tr), "mse_val", np.mean(temp_mse_val))
    best_mse_val = (min(mse_val))
    idx = np.where(mse_val == best_mse_val)
    best_mse_tr = mse_tr[np.squeeze(idx)]
    best_param = params[np.squeeze(idx)]

    cross_validation_visualization(method, params, mse_tr, mse_val)

    if tuned_param == "lambda":
        mse_tr_final, _ = apply_method(method, y, x, x_te = x_te, id = id, lambda_ = best_param, initial_w = initial_w, max_iters = max_iters, gamma = gamma, validation = False)
    elif tuned_param == "gamma":
        mse_tr_final, _ = apply_method(method, y, x, x_te = x_te, id = id, gamma = best_param, lambda_ = lambda_, initial_w = initial_w, max_iters = max_iters, validation = False)
    elif tuned_param == "max_iters":
        mse_tr_final, _ = apply_method(method, y, x, x_te = x_te, id = id, max_iters = best_param, lambda_ = lambda_, initial_w = initial_w, gamma = gamma, validation = False)

    #mse_tr_final, _ = apply_method(method, y, x, np.zeros_like(y), np.zeros_like(x), x_te, id, best_param, validation = False)
    print("final training mse", mse_tr_final)
    print("Chosen " + tuned_param + " is: ", best_param)

    return best_param, best_mse_val, best_mse_tr

def best_triple_param_selection(method, y,x, x_te, id, k_fold, lambdas = [0.1, 0.5], gammas =[0.1,0.5], maxs_iters = [5,10], initial_w = None, seed = 1, verbose = True):
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
        best_mse : scalar, the associated root mean squared error for the best lambda
    """
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    super_best_mse_val = []
    best_gammas = []
    super_best_lambdas = []

    for max_iters in maxs_iters:
        if verbose:
            print("looping for max iters:", max_iters)
        best_mse_val = []
        best_lambdas = []
        for gamma in gammas:
            if verbose:
                print("looping for gamma:", gamma)
            mse_val = []
            for lambda_ in lambdas:
                if verbose:
                    print("looping for lambda:", lambda_)
                temp_mse_val = []
                for k in range(k_fold):
                    loss_tr, loss_val= cross_validation(method, y , x, k_indices, k, lambda_ = lambda_, initial_w = initial_w, max_iters = max_iters, gamma = gamma)
                    temp_mse_val.append(loss_val)
                mse_val.append(np.mean(temp_mse_val))
            best_temp_mse = min(mse_val)
            best_mse_val.append(best_temp_mse)
            best_lambdas.append(lambdas[np.argmin(mse_val)])
        best_mse = min(best_mse_val)
        best_lambda = best_lambdas[np.argmin(best_mse_val)]
        super_best_lambdas.append(best_lambda)
        best_gammas.append(gammas[np.argmin(best_mse_val)])
        super_best_mse_val.append(best_mse)
    super_best_mse = min(super_best_mse_val)
    idx_super_best = np.argmin(super_best_mse_val)
    best_max_iters = maxs_iters[idx_super_best]
    best_gamma = best_gammas[idx_super_best]
    super_best_lambda = super_best_lambdas[idx_super_best]
        
    print("lambda = ", super_best_lambda, "max_iters = ", best_max_iters, "gamma = ", best_gamma, "mse_val = ", super_best_mse)

    #cross_validation_visualization(params, mse_tr, mse_val)

    mse_tr_final = apply_method(method, y, x, x_te = x_te, id = id, max_iters = best_max_iters, lambda_ = best_lambda, initial_w = initial_w, gamma = best_gamma, validation = False)

    #mse_tr_final, _ = apply_method(method, y, x, np.zeros_like(y), np.zeros_like(x), x_te, id, best_param, validation = False)
    print("final training mse", mse_tr_final)

    return best_lambda, best_gamma, best_max_iters, best_mse_val, mse_tr_final


def best_lambda_and_maxiters_selection(method, y, x, x_te, max_iters, k_fold, lambdas, seed = 1):
    """cross validation over regularisation parameter lambda and degree.
    
    Args:
        degrees: shape = (d,), where d is the number of degrees to test 
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_mse : value of the mse for the couple (best_degree, best_lambda)
        
    """
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation over degrees and lambdas: TODO
    #mse_tr = [] isn't useful
    best_mses_te = []
    best_lambdas = []
    for max_iter in max_iters: 
        mse_te = []
        for lambda_ in lambdas:
            #temp_mse_tr = []
            temp_mse_te = []
            for k in range(k_fold):
                loss_tr, loss_te, w_tr = cross_validation(method, y, x, x_te, k_indices, k, lambda_, max_iter)
                #temp_mse_tr.append(loss_tr)
                temp_mse_te.append(loss_te)
            #temp_lambda_mse_tr.append(np.mean(temp_mse_tr))
            mse_te.append(np.mean(temp_mse_te))
            
        best_temp_mse = min(mse_te)
        best_mses_te.append(best_temp_mse) 
        best_lambdas.append(lambdas[np.argmin(temp_mse_te)])
        #mse_tr.append(np.mean(temp_lambda_mse_tr))
    best_mse = (min(best_mses_te))
    best_lambda = best_lambdas[np.argmin(best_mses_te)]
    best_max_iter = max_iters[np.argmin(best_mses_te)]
    # ***************************************************
    #raise NotImplementedError    
    
    return best_max_iter, best_lambda, best_mse