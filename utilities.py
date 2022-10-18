import numpy as np

def compute_mse(y, tx, w):
    """Compute the loss using MSE

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.dot(tx,w)
    loss = 1/(2*y.shape[0])*(np.dot(np.transpose(e),e))
    return loss

def compute_rmse(mse):
    """Compute the rmse given the mse

    Args:
        mse: mean square errors

    Returns:
        The root mean square error
    """
    return np.sqrt(2 * mse)

def compute_gradient_MSE(y, tx, w):
    """Computes the gradient at w of the MSE for linear regression.

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

def sigmoid(t):
    """Apply sigmoid function on t.

    Arg:
        t:
    
    Returns:
        The sigmoid corresponding to the input t 
    """
    return 1/(1 + np.exp(-t))

def compute_loss_neg_loglikeli(y, tx, w):
    """Compute the cost by negative log likelihood.
    
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.
    Returns: 
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    sig = sigmoid(tx.dot(w))
    loss = - y.T@(np.log(sig)) - (1-y).T@(np.log(1 - sig))
    return np.squeeze(loss) # squeeze remove axes of length 1 from loss

def compute_gradient_neg_loglikeli(y, tx, w):
    """Compute the gradient of loss (negative log likelihood).
        Args:
            y: shape=(N, )
            tx: shape=(N,D)
            w: shape=(D, ). The vector of model parameters.
        Returns: 
            the value of the gradient corresponding to the input parameters.
    """
    return tx.T@(sigmoid(tx@(w))-y)/(y.shape[0]) 
    #TODO testé (1/N) selon ce que j'ai lu sur un site mais comprendre pk ça marche 
    #https://medium.com/@IwriteDSblog/gradient-descent-for-logistics-regression-in-python-18e033775082


""""
if __name__ == "__main__":
    assert # Test 
"""





