from curses import use_default_colors
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
        seed: integer.
        
    Returns:
        x_tr: numpy array containing the train data.
        x_val: numpy array containing the validation data.
        y_tr: numpy array containing the train labels.
        y_val: numpy array containing the validation labels.
        
    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    len_tr = int(np.floor(ratio * len(x)))
    data = np.column_stack((x,y))
    data = np.random.permutation(data)
    x_tr = data[0:len_tr,0]
    x_val= data[len_tr::,0]
    y_tr = data[0:len_tr,1]
    y_val= data[len_tr::,1]
    return (x_tr, x_val, y_tr, y_val)

def load_data(path_dataset):
    """load data given the path of the csv file."""
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1)

    label = np.genfromtxt(
        path_dataset, delimiter=",", skip_header=1, usecols=[1],
        converters={1: lambda x: 0 if b"b" in x else 1}) #signal and background ?
    x = data[:,2:-1]
    print(data.shape)
    print("x:", x)
    y = label
    print("y: ", y)
    return x, y

