# -*- coding: utf-8 -*-

import numpy as np
import itertools


def split_data(x, y, ratio, random_enabled=False, start_test=0):
    """ Split the dataset based on the split ratio.
    In this function, we just take the first ratio*n entries 
    as the training set and the rest as the testing test.
    
    It's mostly used when trying to clean the data to remove 
    the random part. 
    
    Parameters
    ----------
    x: array, ndarray like
        Data x
    y: array, ndarray like
        Data y
    ratio: float
        Ration in which to split the data to train and test parts (values in range 0..1). The specified
        value will represent the proportion of training data.
    random_enabled: boolean
        If True, then random samples would be taken as training data and other part will be used as test data.
        If False, then you can specify start index using `start_test` where test samples start, and the
        samples will be gathered sucessevely.

    Returns
    -------
    x_train: array, ndarray like
        Data x, part for training
    y_train: array, ndarray like
        Data y, part for training
    x_test: array, ndarray like
        Data x, part for testing
    y_test: array, ndarray like
        Data y, part for testing
    """
        
    n = len(x)
    if len(y) != n:
        raise ValueError("Vector x and y have a different size")
        
    n_train = int(ratio*n)
    
    if random_enabled:
        # train indices
        train_ind = np.random.choice(n, n_train, replace=False)
        # test indices
        index = np.arange(n)
        mask = np.in1d(index, train_ind)
        test_ind = np.random.permutation(index[~mask])
    else:
        # test indices
        n_test = n - n_train
        test_ind = np.arange(start=start_test, stop=start_test+n_test)
        # train indices
        index = np.arange(n)
        mask = np.in1d(index, test_ind)
        train_ind = index[~mask]
    
    x_train = x[train_ind]
    y_train = y[train_ind]
    
    x_test = x[test_ind]
    y_test = y[test_ind]
    
    return x_train, y_train, x_test, y_test   

def compute_cost(y, tx, w, loss_str):
    """ Computing the cost value
    Developer is also able to specify the way the cost value will be calculated (MSE, MAR, etc.).

    Parameters
    ----------
    y: array, ndarray like
        Data y
    tx: array, ndarray like
        Data x
    w: array, ndarray like
        Weights to be implemented
    loss_str: string
        Options implemented are: MSE, RMSE, MAE, other(used in `ridge_regression`)

    Returns
    -------
    : object
        Cost value
    
    Raises
    ------
    ValueError: The loss provided {loss_str} does not exist. You have the choice between MSE, RMSE, MAE, other
    """
    # Compute the error
    e = y - tx.dot(w)

    if loss_str == "MSE":
        return 1/2*np.mean(e**2)
    elif loss_str == "RMSE":
        return np.sqrt(np.mean(e**2))
    elif loss_str == "MAE":
        return np.mean(np.abs(e))
    elif loss_str == "other":
        return e.dot(e) / (2 * len(e))
    else:
        raise ValueError(f"The loss provided {loss_str} does not exist. You have the choice between MSE, RMSE, MAE, other")

def compute_gradient(y, tx, w, loss_str):
    """ Compute the gradient.
    Developer is also able to specify the way the gradient value will be calculated (MSE, MAR, etc.).

    Parameters
    ----------
    y: array, ndarray like
        Data y
    tx: array, ndarray like
        Data x
    w: array, ndarray like
        Weights to be implemented
    loss_str: string
        Options implemented are: MSE or MAE

    Returns
    -------
    : object
        Gradient value
    
    Raises
    ------
    ValueError: The loss provided {loss_str} does not exist. You have the choice between MSE or MAE
    """
    e = y - np.dot(tx, w)
    N = len(y)

    if loss_str == "MSE":
        return -1/N * np.dot(tx.T, e)
    elif loss_str == "MAE":
        ret_val = 0
        for i in range(len(e)):
            ret_val = ret_val + np.sign(e[i])   # np sign puts 0 when e[i]=0
        return 1/N*ret_val*w
    else:
        raise ValueError(f"The loss provided {loss_str} does not exist. You have the choice between MSE or MAE")        
        
def compute_stoch_gradient(y, tx, w, batch_size, loss_str):
    """ Compute a stochastic gradient for batch data
    Developer is also able to specify the way the stoch gradient value will be calculated (MSE, MAR).

    Parameters
    ----------
    y: array, ndarray like
        Data y
    tx: array, ndarray like
        Data x
    w: array, ndarray like
        Weights to be implemented
    loss_str: string
        Options implemented are located in `compute_gradient`: MSE or MAE

    Returns
    -------
    : object
        Stoch gradient value
    """
    stoch_grad = np.zeros(len(tx[0]))
 
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        stoch_grad = stoch_grad + compute_gradient(minibatch_y, minibatch_tx, w, loss_str)
        
    return 1/batch_size * stoch_grad

def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """ Generating a minibatch iterator for a dataset
    Creating a python generator for Batch iteration.

    Parameters
    ----------
    y: array, ndarray like (iterable)
        Output desired values
    tx: array, ndarray like (iterable)
        Input data
    batch_size: int
        Size of the batch
    num_batches: int
        Number of the batches. Default None
    shuffle: boolean
        Turn ON/OFF the shufflingDefault True.

    Returns
    -------
    iterator: 
        Iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
        Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size / batch_size))
    if not num_batches:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def build_poly(x, degree, crossterm=True, full_crossterm=False, crossterm_squared=False):
    """ Polynomial basis functions
    Polynomial basis functions for input data x, for 0 up to degree degree.

    Parameters
    ----------
    x: array, ndarray like (iterable)
        Input data
    degree: int
        Degree of the polynomial
    crossterm: boolean
        True if you want to add cross term column to product
    crossterm_squared: boolean
        Square product or not

    Returns
    -------
    poly: array, ndarray like
        Polynomioal
    """
    poly = np.ones((len(x), 1))

    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]

    # Add cross term column product CT
    if crossterm:
        temp_col = x[:, 0]
        for j in range(1, x.shape[1]):
            temp_col = np.multiply(temp_col,x[:, j])

        poly = np.c_[poly, temp_col.T]
        poly = np.c_[poly, np.power(temp_col, 2).T]

    if full_crossterm:
        for j in range(x.shape[1]):
           temp_col = np.transpose([x[:,j]])
           temp_matrix = x[:, j+1: -1]
           product = np.multiply(temp_matrix, temp_col)
           poly = np.hstack((poly, product))
           if crossterm_squared:
               poly = np.hstack((poly, np.power(product, 2)))

    return poly

def build_poly_test(x, degree):
    """ Polynomial basis functions withtout cross term support

    Parameters
    ----------
    x: array, ndarray like (iterable)
        Input data
    degree: int
        Degree of the polynomial

    Returns
    -------
    poly: array, ndarray like
        Polynomioal
    """
    n_x = len(x)
    nbr_param = len(x[0])
    poly = np.zeros((n_x, (degree)*nbr_param))
        
    for j in range(nbr_param):
        for k in range(1,degree):
            poly[:, j*(degree)+k] = x[:, j]**k
			
    poly = np.hstack((np.ones((x.shape[0],1)), poly))
            
    return poly


### METHODS IMPLEMENTED ###

def least_square(y, tx):
    """ Calculate the least squares solution
    By default we use RMSE for computing the cost value (loss).

    Parameters
    ----------
    y: array, ndarray like (iterable)
        Output desired values
    tx: array, ndarray like (iterable)
        Input data

    Returns
    -------
    loss: float
        Loss, cost value - minimum loss
    w: array, ndarray like
        Weights - best weights that give minimum loss
    """    
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = compute_cost(y, tx, w, 'RMSE')
    
    return loss, w

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent

    Parameters
    ----------
    y: array, ndarray like (iterable)
        Output desired values
    tx: array, ndarray like (iterable)
        Input data
    initial_w: array, ndarray like
        Initial values of weights
    max_iters: int
        Limitation of iterations
    gamma :
        Gamma from equation

    Returns
    -------
    loss: float
        Loss, cost value - minimum loss
    w: array, ndarray like
        Weights - best weights that give minimum loss
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    iterations = []

    last_loss = 0

    for n_iter in range(max_iters):
        # Compute the gradient and the loss
        loss = compute_cost(y, tx, w, "MAE")
        grad = compute_gradient(y, tx, w, 'MAE')

        # Update w by gradient
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)

        if n_iter % 100 == 0:
            print(f"  Iter={n_iter}, loss={loss}, diff={loss - last_loss}")
            last_loss = loss

            # Stopping criteria for the convergence
        if n_iter > 1 and np.abs(losses[-1] - losses[-2]) < 10 ** -8:
            break

    print(f"  Iter={n_iter}, loss={loss}, diff={loss - last_loss}")
    # Get the latest loss and weights
    return losses[-1], ws[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stohastic gradient descent

    Parameters
    ----------
    y: array, ndarray like (iterable)
        Output desired values
    tx: array, ndarray like (iterable)
        Input data
    initial_w: array, ndarray like
        Initial values of weights
    max_iters: int
        Limitation of iterations
    gamma : float
        Gamma from equation, step size

    Returns
    -------
    loss: float
        Loss, cost value - minimum loss
    w: array, ndarray like
        Weights - best weights that give minimum loss
    """
    # Define a batch size of 500 for the submission
    batch_size = 500

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    iterations = []

    last_loss = 0

    for n_iter in range(max_iters):
        # Compute the stochastic gradient and the loss (See helpers.py for the functions)
        loss = compute_cost(y, tx, w, "MAE")
        grad = compute_stoch_gradient(y, tx, w, batch_size, "MAE")

        # Update w by gradient
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)

        if n_iter % 100 == 0:
            print(f"  Iter={n_iter}, loss={loss}, diff={loss - last_loss}")
            last_loss = loss

        # Stopping criteria for the convergence
        if n_iter > 1 and np.abs(losses[-1] - losses[-2]) < 1e-10:
            break

    print(f"  Iter={n_iter}, loss={loss}, diff={loss - last_loss}")
    
    # Get the latest loss and weights
    return losses[-1], ws[-1]

def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations

    Parameters
    ----------
    y: array, ndarray like (iterable)
        Output desired values (predictions)
    tx: array, ndarray like (iterable)
        Input data (samples)
    lambda_ :
        Lambda from equation

    Returns
    -------
    loss: float
        Loss, cost value - minimum loss
    w: array, ndarray like
        Weights - best weights that give minimum loss
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)

    return compute_cost(y, tx, w, loss_str="other"), w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent
    Use the Logistic Regression method to find the best weights
    
    Parameters
    ----------
    y: array, ndarray like (iterable)
        Output desired values (predictions)
    tx: array, ndarray like (iterable)
        Input data (samples)
    initial_w: array, ndarray like
        Initial values of weights
    max_iters: int
        Limitation of iterations
    gamma :
        Gamma from equation, step size

    
    Returns
    -------
    loss: float
        Loss, cost value - minimum loss
    w: array, ndarray like
        Weights - best weights that give minimum loss
    """
    def sigmoid(x):
        """ Apply sigmoid function on x"""
        result = x
        result[x > 60] = 1
        result[x < -60] = 0
        result[np.abs(x) < 60] = 1/(1 + np.exp(result[np.abs(x) < 60]))
        
        return result

    def log_exp(x):
        """ Apply sigmoid function on x"""
        result = x
        result[x < 60] = np.log(1 + np.exp(result[x < 60]))
        
        return result

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    iterations = []

    last_loss = 0

    for n_iter in range(max_iters):
        # Gradient descent calculation - do one step of gradient descen using logistic regression
        loss = np.sum(log_exp(np.dot(tx, w))) - np.dot(y.T, np.dot(tx, w))
        grad = np.dot(tx.T, sigmoid(np.dot(tx, w)) - y)
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        iterations.append(n_iter)
        if n_iter % 100 == 0:
            print(f"  Iter={n_iter}, loss={loss}, diff={loss - last_loss}")
            last_loss = loss

        # Stopping criteria for the convergence
        if n_iter > 1 and np.abs(losses[-1] - losses[-2]) < 1e-8:
            break

    print(f"  Iter={n_iter}, loss={loss}, diff={loss - last_loss}")
    return losses[-1], ws[-1]