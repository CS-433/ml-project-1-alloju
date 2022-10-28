import numpy as np
import csv

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

def compute_accuracy(y,tx,w):
    e = y - np.dot(tx,w)
    ŷ = np.dot(tx,w)
    ŷ[ŷ < 0] = -1
    ŷ[ŷ >= 0] = 1
    return sum(ŷ != y)/len(y)


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
    #print(gradient)
    return gradient   

def sigmoid(t):
    """Apply sigmoid function on t.

    Arg:
        t:
    
    Returns:
        The sigmoid corresponding to the input t 
    """
    # TODO: handle overflow ! 
    # if t > 100:
    #     sig = 1
    # elif t < -100:
    #     sig = 0
    # else:
    # if t == None:
    #     print("wtf!!")
    #     sig = None
    # else:
    #print("t: ", t.shape)
    ind_over = [t > 100][0]
    t[t > 0]
    #print(ind_over.shape)
    t[ind_over] = 0
    ind_under = [t < -100][0]
    t[ind_under] = 0
    print("indices: ", sum(ind_over), sum(ind_under))
    sig = 1/(1 + np.exp(-t))
    sig[ind_over] = 1
    sig[ind_under] = 0

    return sig

def compute_loss_neg_loglikelihood(y, tx, w):
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
    return (1/y.shape[0])*np.squeeze(loss) # squeeze remove axes of length 1 from loss

def compute_gradient_neg_loglikelihood(y, tx, w):
    """Compute the gradient of loss (negative log likelihood).
        Args:
            y: shape=(N, )
            tx: shape=(N,D)
            w: shape=(D, ). The vector of model parameters.
        Returns: 
            the value of the gradient corresponding to the input parameters.
    """
    print("sig")
    sig = sigmoid(np.dot(tx,w))
    print("grad")
    b = np.dot(tx.T,sig-y)

    return np.dot(tx.T,sigmoid(np.dot(tx,w))-y)/y.shape[0]
    #TODO testé (1/N) selon ce que j'ai lu sur un site mais comprendre pk ça marche 
    #https://medium.com/@IwriteDSblog/gradient-descent-for-logistics-regression-in-python-18e033775082


""""
if __name__ == "__main__":
    assert # Test 
"""

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


