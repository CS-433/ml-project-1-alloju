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

def standardize(x):
    """Standardize the original data set.
    Args:
        x:      input table
    Returns: 
        x:      the standardize input table
        mean_x: the mean vector of the input table
        std_x:  the standard deviation vector of the input table
    """
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def load_csv_title(data_path): 
    """Load the name of each column of the csv file
    
    Args:
        data_path: Path to the file
    
    Returns:
        title: Vector containing the column names
    """
    title = np.genfromtxt(data_path, delimiter=",", dtype = str, max_rows=1)
    title = title[2::]
    return title

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    Args:
        y:              the tabel vector
        tx:             the features table
        batch_size:     size of the batches
        num_batches:    number of batches
        shuffle:        boolean; indicates if the data is randomly shuffled
    Returns:
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer seed
        
    Returns:
        x_tr: numpy array containing the train data.
        x_val: numpy array containing the validation data.
        y_tr: numpy array containing the train labels.
        y_val: numpy array containing the validation labels.
    """
    # set seed
    np.random.seed(seed)
    len_tr = int(np.floor(ratio * len(x)))
    data = np.column_stack((x,y))
    data = np.random.permutation(data)
    x_tr = data[0:len_tr,0:-1] #-1 not included !
    x_val= data[len_tr::,0:-1]
    y_tr = data[0:len_tr,-1]
    y_val= data[len_tr::,-1]
    return (x_tr, x_val, y_tr, y_val)

def load_test_data(path_testset):
    """Load test data and return id and features

    Args:
        path_testset:   relative path to test data

    Returns:
        id:              index of the features
        x:                features
    """
    data = np.genfromtxt(path_testset, delimiter=",", skip_header=1)
    id = data[:,0]
    x = data[:,2:]
    return id, x