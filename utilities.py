import numpy as np

def compute_accuracy(y,tx,w, logistic = False):
    """Compute the accuracy

    Args:
        y:          shape=(N, ), the label vector
        tx:         shape=(N,D), the feature data set
        w:          shape=(D,). The vector of the weight
        logistic:   boolean; specified the type of method

    Returns:
        the accuracy of the model 
    """

    ŷ = np.dot(tx,w)
    if logistic:
        ŷ[ŷ >= 0.5] = 1
        ŷ[ŷ < 0.5] = 0
    else:
        ŷ[ŷ >= 0] = 1
        ŷ[ŷ < 0] = -1
    return sum(ŷ == y)/len(y)

def compute_mse(y, tx, w):
    """Compute the loss using MSE

    Args:
        y:      shape=(N, ), the label vector
        tx:     shape=(N,D), the feature data set
        w:      shape=(D,). The vector of the weight

    Returns:
        loss:   the value of the loss (a scalar), corresponding to the input parameters w.
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
        y:          shape=(N, ), the label vector    
        tx:         shape=(N,D), the features dataset
        w:          shape=(D, ). The vector of weight

    Returns:
        gradient : An array containing the gradient of the loss at w.
    """
    e = y - np.dot(tx,w)
    gradient = -1/y.shape[0] * np.dot(np.transpose(tx),e)
    return gradient   

def sigmoid(t):
    """Apply sigmoid function on t.

    Arg:
        t:   the value for the sigmoid
    
    Returns:
        sig: the sigmoid corresponding to the input t 
    """
    #To avoid overflow
    ind_over = [t > 100][0]
    t[t > 0]
    t[ind_over] = 0
    ind_under = [t < -100][0]
    t[ind_under] = 0

    sig = 1/(1 + np.exp(-t))
    sig[ind_over] = 1
    sig[ind_under] = 0

    return sig

def compute_loss_neg_loglikelihood(y, tx, w):
    """Compute the cost by negative log likelihood.
    
    Args:
        y:          shape=(N, ), the label vector    
        tx:         shape=(N,D), the features dataset
        w:          shape=(D, ). The vector of weight
    Returns: 
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    sig = sigmoid(tx.dot(w))
    loss = - y.T@(np.log(sig)) - (1-y).T@(np.log(1 - sig))
    return (1/y.shape[0])*np.squeeze(loss) 

def compute_gradient_neg_loglikelihood(y, tx, w):
    """Compute the gradient of loss (negative log likelihood).
        Args:
        y:     shape=(N, ), the label vector    
        tx:    shape=(N,D), the features dataset
        w:     shape=(D, ). The vector of weight
        Returns: 
            the value of the gradient corresponding to the input parameters.
    """
    return np.dot(tx.T,sigmoid(np.dot(tx,w))-y)/y.shape[0]
