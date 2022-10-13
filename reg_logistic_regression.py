import utilities as ut 
import numpy as np

def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
 # Logistic regression with a penality (ridge regression)
    w = initial_w
    loss = ut.calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w)) 

    for i in range(max_iters):
        # compute the gradient: 
        gradient = ut.calculate_gradient(y, tx, w) + 2*lambda_*w
        # update w: 
        w -= gamma * gradient
        # compute the cost: 
        loss = ut.calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w)) 

    return w, loss
    

