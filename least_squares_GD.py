import numpy as np
from utilities import compute_mse

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - np.dot(tx,w)
    gradient = -1/y.shape[0] * np.dot(np.transpose(tx),e)
    return gradient


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: the loss value (scalar) for last iteration of GD
        w: the last computed model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        loss = compute_mse(y, tx, w)
        w = w - gamma * grad

    return w, loss