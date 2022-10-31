import numpy as np
import utilities as ut
import helpers as hp

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
    Args:
        y: shape=(N, ), N is the number of samples.
        tx: shape=(N,D), D(= 2 for a linear regression) is the number of features
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last computed model parameters as numpy arrays of shape (D, ), for each iteration of GD
        loss: the loss value (scalar) for last iteration of GD
    """
    w = initial_w
    loss = ut.compute_mse(y, tx, w)
    for n_iter in range(max_iters):
        grad = ut.compute_gradient_MSE(y,tx,w)
        w -= gamma * grad
    loss = ut.compute_mse(y, tx, w)
    return w, np.squeeze(loss)

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent

    Args:
        y: shape=(N, ), N is the number of samples.
        tx: shape=(N,D), D(= 2 for a linear regression) is the number of features
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the last computed model parameters as numpy arrays of shape (D, ), for each iteration of SGD
        loss: the loss value (scalar) for the last iteration of SGD
    """
    w = initial_w
    for minibatch_y, minibatch_tx in hp.batch_iter(y, tx, 1):
        for n_iter in range(max_iters):
            grad = ut.compute_gradient_MSE(minibatch_y,minibatch_tx,w)
            w -= gamma * grad
        loss = ut.compute_mse(minibatch_y, minibatch_tx, w)
    return w, np.squeeze(loss)

def least_squares(y, tx):
    """Least squares regression using normal equations
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: the optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the mse value (scalar) corresponding to the optimal weights
    """
    w = np.linalg.solve(np.dot(tx.T,tx), tx.T@y)
    loss = ut.compute_mse(y,tx,w)
    return w, np.squeeze(loss)

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: the tradeoff parameter (scalar).
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the mse value (scalar) corresponding to the optimal weights
    """
    w = np.linalg.solve(tx.T@tx + 2 * tx.shape[0]*lambda_*np.identity(tx.shape[1]), tx.T@y)
    loss = ut.compute_mse(y, tx, w)
    return w, np.squeeze(loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y ∈ {0, 1})

    Args:
        y: shape=(N, ), N is the number of samples.
        tx: shape=(N,D), D is the number of features.
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: the total number of iterations (scalar) of SGD
        gamma: the stepsize (scalar)

    Returns:
        w: the last computed model parameters as numpy arrays of shape (D, )
        loss: the loss value (scalar) for the last iteration 
    """

    w = initial_w

    for i in range(max_iters):
        # compute the gradient: 
        gradient = ut.compute_gradient_neg_loglikelihood(y, tx, w)
        # update w: 
        w -= gamma * gradient
        # compute the cost: 
    loss = ut.compute_loss_neg_loglikelihood(y,tx,w)
    return w, np.squeeze(loss)

def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    """# Regularized logistic regression using gradient descent or SGD (y ∈ {0, 1}, with regularization term λ∥w∥2 (ridge regulation))

    Args:
        y: shape=(N, ), N is the number of samples.
        tx: shape=(N,D), D is the number of features.
        lambda_: the regularization parameter (scalar)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: the total number of iterations (scalar) of SGD
        gamma: the stepsize (scalar)

    Returns:
        w: the last computed model parameters as numpy arrays of shape (D, )
        loss: the loss value (scalar) for the last iteration 
    """

    w = initial_w
    for i in range(max_iters):
        # compute the gradient with the penalty term: 
        gradient = ut.compute_gradient_neg_loglikelihood(y, tx, w) + 2*lambda_*w
        # update w: 
        w -= gamma * gradient
        # compute the cost: 
    loss = ut.compute_loss_neg_loglikelihood(y,tx,w)
    return w, np.squeeze(loss)