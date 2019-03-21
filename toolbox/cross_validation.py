import numpy as np
from toolbox.implementations import ridge_regression, compute_cost, build_poly
from toolbox.plots import cross_validation_visualization

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # form data with polynomial degree

    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)


    # ridge regression
    loss_tr,w = ridge_regression(y_tr, tx_tr, lambda_)
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_cost(y_tr, tx_tr, w, "MSE"))
    loss_te = np.sqrt(2 * compute_cost(y_te, tx_te, w, "MSE"))
    return loss_tr, loss_te,w

def cross_validation_demo(y,x, degree = 2, k_fold = 4):
    seed = 12

    lambdas = np.logspace(-6, 0, 30)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation
    for lambda_ in lambdas:
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
           loss_tr, loss_te,_ = cross_validation(y, x, k_indices, k, lambda_, degree)
           rmse_tr_tmp.append(loss_tr)
           rmse_te_tmp.append(loss_te)

        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)

def best_degree_selection(y,x, degrees, k_fold, lambda_, seed = 1):
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    #for each degree, we compute the best lambdas and the associated rmse
    best_rmses = []
    best_rmses_tr = []
    #vary degree
    for degree in degrees:
        # cross validation
        rmse_te_tmp = []
        rmse_tr_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te,_ = cross_validation(y, x, k_indices, k, lambda_, degree)
            rmse_te_tmp.append(loss_te)
            rmse_tr_tmp.append(loss_tr)

        best_rmses.append(np.mean(rmse_te_tmp))
        best_rmses_tr.append(np.mean(rmse_tr_tmp))

    ind_best_degree =  np.argmin(best_rmses)
    ind_best_degree_tr =  np.argmin(best_rmses_tr)

    return degrees[ind_best_degree], degrees[ind_best_degree_tr]

def standardize(x, mean_x=None, std_x=None):
    """ Standardize the original data set. """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]


    return x, mean_x, std_x
