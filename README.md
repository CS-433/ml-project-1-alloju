# Machine Learning Project 1 

## Install

- To clone the repository, write in the terminal : `git clone https://github.com/CS-433/ml-project-1-alloju.git`
- To install the environment, please use the environment.yml file provided at `https://github.com/epfml/ML_course/tree/master/projects/project1/grading_tests` <br/>
    Then run: <br/>
    `conda create --file=environment.yml --name=project1-grading` <br/>
    `conda activate project1-grading`
- Put the datasets found here: `https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/dataset_files` in a folder called `data`

## Reproducibility 

To reproduce the results presented in the report, the user should run the following command in its terminal `python3 run.py`. <br/>
This produces a csv file that contains our best prediction posted on AICrowd. <br/>
To reproduce the rest of the results, the user should open `run.py` and decoment all the lines and run the same command. <br/>
This reproduces the accuracy values for each implementation presented in the report.

## Implementations

In the file `implementations.py`, the user can find the following required implementations :

- `least_squares` : the least squares error implementation, returning the weights and loss using mean squared error
- `mean_squared_error_sgd` : the mean squared error using stochastic gradient descent, returning the weights and loss using mean squared error
- `mean_squared_error_gd`: the mean squared error using gradient descent, returning the weights and loss using mean squared error
- `ridge_regression` : the mean squared implementation with L2 regularization, returning the weights and loss using mean squared error
- `logistic_regression` : the logistic regression implementation, returning the weights and loss using negative log likelyhood
- `reg_logistic_regression` : the L2 regularized logistic regression, returning the weights and loss using negative log likelyhood

## Preprocessing

There are two files regarding the preprocessing. 

- First `prepro_plots.ijynb` containing the visualisation of the data. By running this command in the terminal `python3 prepro_plots.ipynb` the user can observe some basic plots, allowing to get a better vision of the data.
- Secondly `preprocessing.py` containing all the different steps implemented in the preprocessing as well as the two final preprocessing functions, one for the train and one for the test set.

## Useful files 

- In `paths.py` the user can specify the path to their different folders. *E.g. folder containing data, folder in which to put the predictions,...*
- In `helpers.py` the user can find all the methods that were given at the beginning of the project. 
- In `utilities.py` the user can find methods that have been used accross the program and were stored here for easy access.
- In `paths.py` we stored all the different paths allowing us to access and store our data and prediction files.
- In `plots.py` one can find the methods that allowed us to plot and visualize our results.
- In `methods.py` the user will find the modular method to call any implementation and the predict method that creates submission files. We also stored here the cross validation implementation. This means the generic functions splitting the data and applying cross validation, but mostly one will find the implemented methods that can determine the best single or triple param for the different implementations, as well as the selection of the best degree for polynomial feature expansion.

## Tests 

Please use the provided tests from the course repository. Clone the course repository and run them from this directory: <br/>
`pytest --github_link <GITHUB-REPO-URL> . `


