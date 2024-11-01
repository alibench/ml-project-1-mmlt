import numpy as np
from utils import *

# -----------------------------------------IMPLEMENTATIONS----------------------------------------------------#

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression algorithm using gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D+1)
        initial_w: numpy array of shape=(D+1, ). The initialization for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize.
        
    Returns:
        loss: loss value (scalar), corresponding to the input parameters w
        w: model parameters as numpy array of shape (D+1, ).
    """
    
    # Initialize the model parameters
    w = initial_w
    
    # Iterate over the number of max iterations
    for n_iter in range(max_iters):
        # Compute the gradient with respect to w
        gradient = compute_gradient_mse(y, tx, w)
        
        # Update the model parameters
        w = w - gamma * gradient
        
    # Compute the final loss after all iterations
    loss = compute_loss_mse(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression algorithm using stochastic gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D+1)
        initial_w: numpy array of shape=(D+1, ). The initialization for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize.
        
    Returns:
        loss: loss value (scalar), corresponding to the input parameters w
        w: model parameters as numpy array of shape (D+1, ).
    """
    
    # Initialize the model parameters
    w = initial_w
    
    # Number of samples N
    N = len(y)
    
    # Iterate over the number of max iterations
    for n_iter in range(max_iters):
        # Select a random data sample i among the N
        i = np.random.randint(0, N)
        
        # Compute the gradient with respect to w using the single data sample i
        gradient = compute_gradient_mse(np.array([y[i]]), np.array([tx[i]]), w)
        
        # Update the model parameters
        w = w - gamma * gradient
        
    # Compute the final loss after all iterations
    loss = compute_loss_mse(y, tx, w)

    return w, loss
    
    
def least_squares(y, tx):
    """Least squares regression using normal equations.

    Args:
        y: numpy array of shape (N, )
        tx: numpy array of shape (N, D+1)
        
    Returns:
        w: optimal weights, numpy array of shape(D+1,)
    """
    
    # Compute the optimal weights using the normal equations
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    
    # Compute the MSE loss with the optimal weights
    loss = compute_loss_mse(y, tx, w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D+1)
        lambda_: a scalar denoting the regularization (penalty) term
    """
    
    # Compute the optimal weights using the normal equations
    w = np.linalg.solve(tx.T @ tx + lambda_* 2 * y.shape[0] * np.identity(tx.shape[1]), tx.T @ y)
    
    # Compute the MSE loss with the optimal weights
    loss = compute_loss_mse(y, tx, w)
    
    return w, loss
    
    
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression algorithm using gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D+1)
        initial_w: numpy array of shape=(D+1, ). The initialization for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize.
        
    Returns:
        loss: loss value (scalar), corresponding to the input parameters w
        w: model parameters as numpy array of shape (D+1, ).
    """
    
    # Initialize the model parameters
    w = initial_w
    
    # Iterate over the number of max iterations
    for n_iter in range(max_iters):
        # Compute the gradient with respect to w
        gradient = compute_gradient_mle(y, tx, w)
        
        # Update the model parameters
        w = w - gamma * gradient
    
    # Compute the final loss after all iterations
    loss = compute_loss_mle(y, tx, w)
    
    return w, loss
    
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Args:
        y: numpy array of shape (N, )
        tx: numpy array of shape (N, D+1)
        lambda_: Regularization parameter (penalty term)
        initial_w: Initial weights of shape (D+1,)
        max_iters: Number of iterations for gradient descent
        gamma: Step size (learning rate)
        
    Returns:
        w: The optimized weight vector of shape (D+1,)
        loss: The regularized logistic loss (cross-entropy) after all iterations
    """
    
    # Initialize weights
    w = initial_w
    
    # Iterate over the number of max iterations
    for n_iter in range(max_iters):
        # Compute the gradient of the regularized logistic loss
        gradient = compute_gradient_mle(y, tx, w) + 2 * lambda_ * w
        
        # Update the weights
        w = w - gamma * gradient
    
    # Compute the final loss
    loss = compute_loss_mle(y, tx, w)
    
    return w, loss
