# Machine Learning Project 1 

## Install

To clone the repository, copy in the terminal : `git clone https://github.com/CS-433/ml-project-1-alloju.git`
BALABABDJBJKBJKSDBJKSBD sur l'environnement

## Reproducibility 

To reproduce the results presented in the report, the user should run the following command in its terminal `python3 run.py`. This produces a csv file that contains our best prediction posted on AICrowd. 
To reproduce the rest of the results, the user should decoment the lines ... to ... and run the same command. This reproduces the accuracy values for each implementation presented in the report.

## Implementations

In the file `implementations.py`, the user can find the following required implementations :
- `least_squares` : the least square error implementation, returning the weights and loss using mean square error
- `mean_squared_error_sgd` : the least square error using stochastic gradient descent, returning the weights and loss using mean square error
- `mean_squared_error_gd`: the least square error using gradient descent, returning the weights and loss using mean square error
- `ridge_regression` : the least square implementation with L2 regularization, returning the weights and loss using mean square error
- `logistic_regression` : the logistic regression implementation, returning the weights and loss using negative log likelyhood
- `reg_logistic_regression` : the L2 regularized logistic regression, returning the weights and loss using negative log likelyhood

## Preprocessing

There are to files regarding the preprocessing. 
First `prepro_plots.ijynb` containing to visualisation of the data. By running this command in the terminal `python3 prepro_plots.ijynb` the user can observe some basic plots allowing to get a better vision of the data.
Secondly `preprocessing.py` containing all the different steps implemented in the preprocessing as well as the two final preprocessing functions, one for the train and one for the test set.

## Useful files 

- In `helpers.py` the user can find all the methods that were given at the beginning of the project. 
- In `utilities.py` the user can find methods that have been used accross the program and were stored here for easy access.
- In `paths.py` we stored all the different paths allowing us to access and store our data and prediction files.
- In `plots.py` one can find the methods that allowed us to plot and visualize our results.
- In `apply_method.py` the user will find the modular method to call any implementation and the predict method that creates submission files

## Cross validation

- All cross validation implementations are to be found in `cross_validation.py`. This means the generic functions splitting the data and applying cross validation, but mostly one will find the implemented methods that can determine the best single or triple param for the different implementations, as well as the selection of the best degree for polynomial feature expansion. 

## Tests 

BLALBANJSBHDBUEBJHBDJEHBFJH je sais pas quoi dire

