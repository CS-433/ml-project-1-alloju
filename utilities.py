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

def sigmoid(t):
    """Apply sigmoid function on t."""
    return 1/(1 + np.exp(-t))

def loss_neg_log_likelihood(y, tx, w):
    """Compute the cost by negative log likelihood."""
    sig = sigmoid(tx.dot(w))
    loss = - y.T.dot(np.log(sig)) - (1-y).T.dot(np.log(1 - sig))
    return np.squeeze(loss) # squeeze remove axes of length 1 from loss

def gradient_neg_log_likelihood(y, tx, w):
    """Compute the gradient of loss (negative log likelihood)."""
    return tx.T.dot(sigmoid(tx.dot(w))-y)


""""
if __name__ == "__main__":
    assert # Test 
"""





