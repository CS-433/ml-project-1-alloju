from split_data import load_data
import numpy as np
import csv
from helpers import standardize
import matplotlib.pyplot as plt

data_path = "data/train.csv"
x,y = load_data(data_path)

def get_id(data_path):
    id = np.genfromtxt(data_path, delimiter=",", skip_footer=250000, dtype = str)
    return id[2:-1]

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

def plot_hist(x0, x1, id):
    for i in range (4):
        plt.hist(x0.T[i], bins = 100, density = True, label = 'Background', alpha = 0.5)
        plt.hist(x1.T[i], bins = 100, density = True, label = 'Signal', alpha = 0.5)
        plt.legend()
        plt.title(id[i])
        plt.show()
    return None

def boxplot(x):
    for i in range(10, 14, 1):
        dict = plt.boxplot(x.T[i])
        plt.show()
    return None

def remove_outliers(x):
    for i in range(len(x[1])):
        feature_removed = x[:,i][x[:,i] != -999]
        qs = np.quantile(feature_removed, np.array([0.25, 0.5, 0.75]), axis = 0)
        ir = qs[2]-qs[0]
        lower_limit = qs[0] - (1.5*ir)
        upper_limit = qs[2] + (1.5*ir)
        x[:, i][x[:, i] < lower_limit] = qs[1]
        x[:, i][x[:, i] > upper_limit] = qs[1]
        x[:, i][x[:, i] == -999] = qs[1]
    return x

def replace_class(x):
    len_ = np.shape(x)[0]
    #create 4 new colomns with indexes 25 to 28 containing solely zeros
    x = np.insert(x, 30, np.zeros(len_), axis = 1)
    x = np.insert(x, 30, np.zeros(len_), axis = 1)
    x = np.insert(x, 30, np.zeros(len_), axis = 1)
    x = np.insert(x, 30, np.zeros(len_), axis = 1)
    #replace indixes that are missing
    x[:, 30][x[:,29] == 0] = 1
    x[:, 31][x[:,29] == 1] = 1
    x[:, 32][x[:,29] == 2] = 1
    x[:, 33][x[:,29] == 3] = 1
    #delete the colomn
    x = np.delete(x, 29, axis = 1)
    return x

def corr(x):
    #TODO check which attributes have high correlation, if above a certain threshold, delete one of them
    #corr = np.zeros([x.shape[1], x.shape[1]])
    corr = np.corrcoef(x,rowvar = False)
    corr[:][np.isclose(corr,1)] = 0
    corr[:][corr < 0.95] = 0
    corr = np.triu(corr)
    dia = np.diag(corr)
    ind = (np.array([np.nonzero(corr)]))
    return x

def preproc(x):
    x = remove_outliers(x)
    x = angle_values(x)
    x = replace_class(x)
    #x = corr(x)
    x, x_mean, x_std = standardize(x)
    return x

x= preproc(x)
#id = get_id(data_path)
#x0,x1 = separate(x, y)
#plot_hist(x0,x1,id)
#boxplot(x)
