from split_data import load_data
import numpy as np
import csv
from helpers import standardize


#data_path = "data/train.csv"
#x,y = load_data(data_path)

def missing_data(x): # à voir si on utilise mean ou median
    d = x.shape[1] # determine the number of features
    for i in range(d):
        feature_removed = x[:,i][x[:,i] != -999] #remove all features equal to -999 (undetermined)
        mean = np.mean(feature_removed) #determine the mean
        x[:,i][np.isclose(x[:,i],-999)]=mean #replace undetermined values by the mean
    return x

def separate(x,y):
    x_0 = np.array([])
    x_1 = np.array([])
    x_0 = np.append(x_0, np.where(y==-1))
    x_1 = np.append(x_1, np.where(y==1))
    return x_0,x_1

def normalize(x):
    colomns = np.array([15,18,20,25,28]) #index of colomns containing the colomns with angle values --> no normalization
    means = np.mean(x, axis = 0)
    stdev = np.std(x, axis = 0)
    x = np.add(x, - means)
    x = x/stdev
    return x
       
def angle_values(x):
    colomns = np.array([15,18,20,25,28]) #index of colomns containing the colomns with angle values
    for i in range(len(colomns)):
        sin = np.sin(x[:,i]) # compute sin of angle
        cos = np.cos(x[:,i]) # compute cos of angle
        x = np.delete(x, colomns[i], axis = 1) # delete angle value
        x = np.insert(x, i, sin, axis = 1) # insert sin
        x = np.insert(x, i, cos, axis = 1) # insert cos
        colomns = colomns + 1
    return x

def preproc(x):
    x = missing_data(x)
    x = angle_values(x)
    x, x_mean, x_std = standardize(x)
    return x
        
#data =missing_data(x)      
#x0,x1 = separate(x,y)
