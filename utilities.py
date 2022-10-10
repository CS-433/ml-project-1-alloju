import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using MSE

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.dot(tx,w)
    L = 1/(2*y.shape[0])*(np.dot(np.transpose(e),e))
    return L
