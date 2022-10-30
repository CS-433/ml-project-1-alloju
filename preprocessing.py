from split_data import load_data
import numpy as np
import csv
from helpers import load_csv_data, standardize
import matplotlib.pyplot as plt

data_path = "data/train.csv"
x,y, ids = load_csv_data(data_path)

def to_0_1(y):
    """Change the input from {-1;1} to {0;1}
    
    Args:
        y: The vector to change
    
    Returns:
        y: The vector with the changed values
    """
    y_0_1 = np.zeros_like(y)
    y_0_1[y >= 0] = 1
    y_0_1[y < 0] = 0
    return y_0_1


def get_id(data_path): #TODO a enlever ! on en a plus besoin mtn non ? 
    """

    Args:
        data_path: 
    
    Returns:
        id: 
    """
    id = np.genfromtxt(data_path, delimiter=",", skip_footer=250000, dtype = str)
    return id[2:-1]

def missing_data(x): #TODO est ce qu'on en a encore besoin ? 
    """Define columns with too much missing data
    
    Args:
        x: The data table
    
    Returns:
        x: The input table with the value of the selected table changed to 0 
    """
    special = [4,5,6,12,23,24,25,26,27,28]
    for i in special :
        x[:,i][x[:,i] == -999] = 0
    return x


def delete_correlated(x, ind, title):
    """Delete the column of x determined as correlated by the corr(x) function
    
    Args:
        x: The input vector 
        ind: The indice of the column that we want to remove from x
        title: Vector containing the name of the column from x
    
    Returns:
        x: The x input without the correlated column
        title: The title input without the name of the correlated column 
    """
    x = np.delete(x, ind, axis = 1)
    title = np.delete(title, ind)
    return x, title


def separate(x,y): #TODO est ce qu'on a toujours besoin ? si oui prendre en conpte que mtn on a y = {0,1}
    """Separation of the data regarding the prediction value
    
    Args:
        x: The data table
        y: The prediction vector
    
    Returns:
        x_0: The input table with the prediction equal to 1
        x_1: The input table with the prediction equal to -1
    """
    x_0 = np.delete(x, np.where(y==1), axis = 0)
    x_1 = np.delete(x, np.where(y==-1), axis = 0)
    return x_0,x_1

def angle_values(x, title):
    """Transform the angle value into sin and cos
    
    Args:
        x: The input vector 
        title: Vector containing the name of the column from x 
    
    Returns:
        x: The input vector with the new way of angle expression
        title: Vector containing the name of the new columns and without the angle ones
    """
    columns_id = [idx for idx, s in enumerate(title) if 'phi' in s and 'centrality' not in s ]
    columns_id = np.squeeze(columns_id)
    for j in range(len(columns_id)):
        i = columns_id[j]

        sin = np.sin(x[:,i]) # compute sin of angle
        cos = np.cos(x[:,i]) # compute cos of angle
        t = title[i]

        x = np.delete(x, i, axis = 1) # delete angle value
        title = np.delete(title, i)
        x = np.insert(x, i, sin, axis = 1) # insert sin
        title = np.insert(title, i, t +'_sin') # insert title sin
        x = np.insert(x, i, cos, axis = 1) # insert cos
        title = np.insert(title, i, t+'_cos') # insert title cos
        columns_id = columns_id +1
    return x, title


def plot_hist(x0, x1, id):
    """Plot an histogram of the data
    
    Args:
        x_0: The data with y = 1
        x_1: The data with y = -1
        id: The indice of the columns to plot
    
    Returns:
        None 
    """
    for i in range (4):
        plt.hist(x0.T[i], bins = 100, density = True, label = 'Background', alpha = 0.5)
        plt.hist(x1.T[i], bins = 100, density = True, label = 'Signal', alpha = 0.5)
        plt.legend()
        plt.title(id[i])
        plt.show()
    return None

def boxplot(x):
    """Create boxplot of the input values
    
    Args:
        x: The data  
    
    Returns:
        None 
    """
    for i in range(10, 14, 1):
        dict = plt.boxplot(x.T[i])
        plt.show()
    return None

def remove_outliers(x):
    """Remove the outliers by replacing them with with the median 
    
    Args:
        x: The input 
    
    Returns:
        x: The corrected input
    """
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


def replace_class(x, title):
    """
    
    Args:
        x: The input vector 
        title: Vector containing the name of the column from x  
    
    Returns:
        x: The input vector with the separation of the PRI_jet_num values 
        title: Vector containing the name of the new columns
    """
    len_ = np.shape(x)[0]
    column_id = [idx for idx, s in enumerate(title) if 'PRI_jet_num' in s]
    column_id = np.squeeze(column_id)
    ind = np.unique(x[:,column_id])
    for j in reversed(range(len(ind))):
        i = ind[j]
        # Create 4 new columns with indexes column_id to column_id+3 containing solely zeros
        x = np.insert(x, column_id+1, np.zeros(len_), axis = 1)
        title = np.insert(title, column_id+1, title[column_id]+'_'+str(int(i)))
        #replace indixes that are missing
        x[:, column_id+1][x[:, column_id] == i] = 1

    #delete the colomn
    x = np.delete(x, column_id, axis = 1)
    title = np.delete(title, column_id)

    return x, title

def pca(x, percentage, feature_to_keep):
    """Principal component analysis. If both percentage of variance explained and number of features to keep are given,
    the value taken is the number of fature to keep.
    
    Args:
        x: The table of values to whom the PCA is applied 
        percentage: percentage of variance explained wanted
        feature_to_keep: nb of feature we want to keep
    
    Returns:
        projection_matrix: The projection matrix with the eigen vectors
    """
    cov = np.cov(x.T)
    eigen_val, eigen_vec = np.linalg.eig(cov)
    variance_explained = []
    for i in eigen_val:
        variance_explained.append((i/sum(eigen_val))*100)
    cumulative_variance_explained = np.cumsum(variance_explained)
    if feature_to_keep == 0:
        explained_percentage =np.squeeze((cumulative_variance_explained > percentage).nonzero())
        nb_component_keep = explained_percentage[0]
    else:
        nb_component_keep = feature_to_keep
    print('PCA: nb components to keep: ', nb_component_keep)
    projection_matrix = (eigen_vec.T[:][:nb_component_keep]).T
    return projection_matrix

def corr(x):
    """Search for the correlation between the columns of the input. 
    Check which attributes have high correlation, if above a certain threshold, delete one of them 
    
    Args:
        x: Input table
    
    Returns:
        ind_to_delete: The indice of the column that are correlated to others
    """
    #id = get_id(data_path)
    corr = np.abs(np.corrcoef(x,rowvar = False))
    #plt.imshow(corr)
    #plt.colorbar()
    #plt.xticks(ticks=np.arange(len(id)),labels=id,rotation=90)
    #plt.yticks(ticks=np.arange(len(id)),labels=id)
    #plt.show()
    corr = np.triu(corr, k=1)
    ind = np.where(corr > 0.95)
    ind_to_delete = np.unique(ind[0])
    return np.squeeze(ind_to_delete)

def build_poly(x, title, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    #x = np.insert(x, 0, np.ones_like(x[:,0]), axis = 1)
    #title = np.insert(title, 0,"bias")
    for i in range(0, (x.shape[1]) * degree, degree):
        #looping on features
        t = title[i]
        for j in range(2,degree+1):
            x = np.insert(x, i + j-1, x[:,i]**j, axis = 1)
            title = np.insert(title, i + j-1, t +'_x**' + str(j))
    return x

def preproc_train(x, title, percentage = 95, feature_to_keep = 0, do_corr = True, do_pca = True, do_poly = False, degree = 0):
    """Preprocessing for the training data
    #TODO finir de compléter
    Args:
        x: The input table
        title: The name of the x columns
        do_corr: Boolean that indicate the deletion of correlated column is applied 
        do_pca: Boolean that indicate the PCA is applied 
    
    Returns:
        x: 
        x_mean:
        x_std:
        ind: 
        projection_matrix:
    """
    title = title
    if do_corr:
        ind = corr(x)
        x, title = delete_correlated(x, ind, title)
    else:
        ind = None
    x = remove_outliers(x)
    x, title = angle_values(x, title)
    x, title = replace_class(x, title)
    x, x_mean, x_std = standardize(x)
    if do_pca:
        projection_matrix = pca(x, percentage, feature_to_keep)
        x = np.dot(x, projection_matrix)
    else:
        projection_matrix = None
    if do_poly:
        x = build_poly(x, title, degree)
    
    return x, x_mean, x_std, ind, projection_matrix


def preproc_test(x, title, x_mean, x_std, projection_matrix, ind, do_corr = True, do_pca = True, do_poly = False, degree = 0): 
    """Preprocessing for the test data
    #TODO finir de compléter
    Args:
        x: The input table
        title: The name of the x columns
        x_mean:
        x_std:
        projection_matrix:
        ind:
        do_corr: Boolean that indicate the deletion of correlated column is applied 
        do_pca: Boolean that indicate the PCA is applied 
    
    Returns:
        x: 
    """
    title = title
    if do_corr:
        x, title = delete_correlated(x, ind, title)
    x = remove_outliers(x)
    x, title = angle_values(x, title)
    x, title = replace_class(x, title)
    x = x-x_mean
    x = x/x_std
    if do_pca: 
        x = np.dot(x, projection_matrix)
    if do_poly:
        x = build_poly(x, title, degree)
    return x

#id = get_id(data_path)
#x0,x1 = separate(x, y)
#plot_hist(x0,x1,id)
#boxplot(x)
