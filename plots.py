# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import matplotlib.pyplot as plt
from paths import plots_dir
import os

def cross_validation_visualization(method, params, mse_tr, mse_te, tuned_param,i):
    """visualization the curves of mse_tr and mse_te
    Args:
        method:         method to apply to the data
        params:         value(s) of the tuned parameter
        mse_tr:         mse of the training data
        mse_te:         mse of the test data
        tuned_param:    the tuned parameter
        i:              numerotation of the figure
    """
    plt.figure(i)
    plt.semilogx(params, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(params, mse_te, marker=".", color='r', label='test error')
    plt.xlabel(tuned_param)
    plt.ylabel("mse")
    #plt.xlim(1e-4, 1)
    plt.title("cross validation" + str(method))
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir,"cross_validation" + str(method)))

def cross_validation_visualization_degree(method, params, mse_tr, mse_te, tuned_param,i):
    """visualization the curves of mse_tr and mse_te.
    Args:
        method:         method to apply to the data
        params:         value(s) of the tuned parameter
        mse_tr:         mse of the training data
        mse_te:         mse of the test data
        tuned_param:    the tuned parameter
        i:              
    """
    plt.figure(i)
    plt.plot(params, mse_tr, marker=".", color='b', label='train error')
    plt.plot(params, mse_te, marker=".", color='r', label='test error')
    plt.xlabel(tuned_param)
    plt.ylabel("mse")
    #plt.xlim(1e-4, 1)
    plt.title("cross validation" + str(method))
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir,"cross_validation" + str(method)))

def cross_validation_visualization_multiple(method, params, mse_tr, mse_te, tuned_param, additional_param,i):
    """visualization the curves of mse_tr and mse_te.
    Args:
        method:             method to apply to the data
        params:             value(s) of the tuned parameter
        mse_tr:             mse of the training data
        mse_te:             mse of the test data
        tuned_param:        the tuned parameter
        additional_param:   additional param made for degree
        i:              
    """
    plt.figure(i)
    plt.semilogx(params, mse_tr, marker=".", label='train error for degree: ' + str(additional_param))
    plt.semilogx(params, mse_te, marker=".", label='test error for degree: ' + str(additional_param) )
    plt.xlabel(tuned_param)
    plt.ylabel("mse")
    #plt.xlim(1e-4, 1)
    plt.title("cross validation" + str(method))
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir,"cross_validation" + str(method)))