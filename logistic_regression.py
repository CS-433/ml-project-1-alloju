# from utilities import *
import utilities as ut

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    w = initial_w
    loss = ut.calculate_loss(y, tx, w)
    for i in range(max_iters):
        # compute the gradient: 
        gradient = ut.calculate_gradient(y, tx, w)
        # update w: 
        w -= gamma * gradient
        # compute the cost: 
        loss = ut.calculate_loss(y, tx, w)
    return w, loss



