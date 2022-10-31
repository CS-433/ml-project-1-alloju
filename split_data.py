import numpy as np 

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