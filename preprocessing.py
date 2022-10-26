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

def missing_data(x): 
    # define colomns with too much missing data
    special = [4,5,6,12,23,24,25,26,27,28]
    for i in special :
        x[:,i][x[:,i] == -999] = 0
    return x

def delete_correlated(x, ind):
    return np.delete(x, ind, axis = 1)

def separate(x,y):
    x_0 = np.delete(x, np.where(y==1), axis = 0)
    x_1 = np.delete(x, np.where(y==-1), axis = 0)
    return x_0,x_1

def angle_values(x):
    colomns = np.array([11,14,16,19,20]) #index of colomns containing the colomns with angle values
    for j in range(len(colomns)):
        i = colomns[j]
        sin = np.sin(x[:,i]) # compute sin of angle
        cos = np.cos(x[:,i]) # compute cos of angle
        x = np.delete(x, i, axis = 1) # delete angle value
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
    x = np.insert(x, 22, np.zeros(len_), axis = 1)
    x = np.insert(x, 22, np.zeros(len_), axis = 1)
    x = np.insert(x, 22, np.zeros(len_), axis = 1)
    x = np.insert(x, 22, np.zeros(len_), axis = 1)
    #replace indixes that are missing
    x[:, 22][x[:,21] == 0] = 1
    x[:, 23][x[:,21] == 1] = 1
    x[:, 24][x[:,21] == 2] = 1
    x[:, 25][x[:,21] == 3] = 1
    #delete the colomn
    x = np.delete(x, 21, axis = 1)
    return x

def pca(x):
    cov = np.cov(x.T)
    eigen_val, eigen_vec = np.linalg.eig(cov)
    variance_explained = []
    for i in eigen_val:
        variance_explained.append((i/sum(eigen_val))*100)
    cumulative_variance_explained = np.cumsum(variance_explained)
    explained_99 =np.squeeze((cumulative_variance_explained > 99).nonzero())
    nb_component_keep = explained_99[0]
    projection_matrix = (eigen_vec.T[:][:nb_component_keep]).T
    return projection_matrix

def corr(x):
    corr = np.triu(np.abs(np.corrcoef(x,rowvar = False)), k=1)
    ind = np.where(corr > 0.95)
    ind_to_delete = np.unique(ind[0])
    return np.squeeze(ind_to_delete)

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x

def preproc_train(x):
    ind = corr(x)
    x = delete_correlated(x, ind)
    x = remove_outliers(x)
    x = angle_values(x)
    x = replace_class(x)
    x, x_mean, x_std = standardize(x)
    projection_matrix = pca(x)
    #x = np.dot(x, projection_matrix)
    return x, x_mean, x_std, ind, projection_matrix

def preproc_test(x, x_mean, x_std, ind, projection_matrix):
    x = delete_correlated(x, ind)
    x = remove_outliers(x)
    x = angle_values(x)
    x = replace_class(x)
    x = x-x_mean
    x = x/x_std
    #x = np.dot(x, projection_matrix)
    return x, x_mean, x_std

x_tr, x_mean, x_std, ind, projection_matrix  = preproc_train(x)
x_te, x_mean, x_std = preproc_test(x, x_mean, x_std, ind, projection_matrix)
print(x_tr.shape)
print(x_te.shape)

