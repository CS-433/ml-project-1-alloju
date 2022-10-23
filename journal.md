#### 17.10:
- 1st submission: titre manquant
- 2nd submission: 0,1 isn't expected
- 3rd submission: 1st trial output -1,1 -with logistic regression lambda_ = 0.1, initial_w = np.random.rand(x_tr.shape[1]), max_iters = 100, gamma = 0.1 
- 4th submission: 1st trial with leastsquare, output -1,1

#### 18.10
- 1st submission: logistic regression, correction angles then standardize tr_loss = nan, val_loss = 58
- 2nd submission: least squares, correction angles then standardize, tr_loss = 0.825, val_loss = 0.827
- 3rd submission: 1st ridge regression, tr_loss = 0.88, val_loss = 0.88, lambda_ = 0.1
- 4th submission: 1st GD, tr_loss = 1.6, val_loss = 1.62, max_iters = 50, gamma = 0.01)
- 5th submission: 1st SGD, tr_loss = 0.62, val_loss = 2.83, max_iters = 5, gamma = 0.08

## 21.10 
- ??

#### 22.10
- 1st submission: reg_logistic_regression, y, x, x_te, id, 10, lambdas = [0.5,1,3,5,6,7,8,9,10,15,50,80], gammas = [0.01,0.02,0.05,0.1,0.25,0.5,0.75,0.9], maxs_iters = [5,10,20,50,75,100,150,200]) -> chosen: lambda = 7, max_iters = 10, gamma = 0.05, rmse training = 1.137
- 2nd submission: only standardization: reg_log_reg: lambda =  7 max_iters =  10 gamma =  0.05 rmse_tr =  0.9703284017785432
final training rmse (0.9703325472392822, 0)
- 3rd submission: only std: mean squared error GD: lambda =  0.0 max_iters =  10 gamma =  0.05 rmse_tr =  238159413519.04776
final training rmse (0.9229663064909535, 0)
- 4th submission: only std: ridge reg: lambda = 0.0, final training rmse (0.8239500193849315)

#### 23.10
- 1st sumission: only standardization: reg_log_reg: lambda =  7.5 max_iters =  5 gamma =  0.01 rmse_tr =  0.9429446373340948
final training rmse (0.9388746578777089, 0)