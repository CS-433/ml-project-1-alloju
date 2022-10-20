from split_data import load_data
import numpy as np
import csv
from helpers import standardize
import matplotlib.pyplot as plt


data_path = "data/train.csv"
x,y = load_data(data_path)

def missing_data(x): # à voir si on utilise mean ou median
    d = x.shape[1] # determine the number of features
    for i in range(d):
        feature_removed = x[:,i][x[:,i] != -999] #remove all features equal to -999 (undetermined)
        mean = np.mean(feature_removed) #determine the mean
        x[:,i][np.isclose(x[:,i],-999)]=mean #replace undetermined values by the mean
    return x

def separate(x,y):
    x_0 = np.delete(x, np.where(y==1), axis = 0)
    x_1 = np.delete(x, np.where(y==-1), axis = 0)
    return x_0,x_1

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

def plot_hist(x0, x1):
    for i in range (len(x0[0])):
        plt.hist(x0.T[i], bins = 100, density = True, label = 'Background', alpha = 0.5)
        plt.hist(x1.T[i], bins = 100, density = True, label = 'Signal', alpha = 0.5)
        plt.legend()
        plt.show()
    return None

def boxplot(x):
    for i in range(10, 14, 1):
        dict = plt.boxplot(x.T[i])
        plt.show()
    return None

def preproc(x):
    x = missing_data(x)
    x = angle_values(x)
    x, x_mean, x_std = standardize(x)
    return x

x =missing_data(x)
x = angle_values(x) 
x, x_mean, x_std = standardize(x) 
x0,x1 = separate(x, y)
#plot_hist(x0,x1)
boxplot(x)
