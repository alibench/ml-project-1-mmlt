import numpy as np
from utils import *
from metrics import f1_score

# -----------------------------------------IMPLEMENTATIONS----------------------------------------------------#

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression algorithm with gradient descent.

    Parameters:
        y : numpy array of shape=(N, ).
        tx : numpy array of shape=(N, D+1).
        initial_w : numpy array of shape=(D+1, ) corresponding to the initialization for the model parameters.
        max_iters : a scalar denoting the total number of iterations of gradient descent.
        gamma : a scalar denoting the stepsize.
        
    Returns:
        loss: final loss value (scalar).
        w: model parameters as numpy array of shape (D+1, ).
    """
    w = initial_w
    
    for _ in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
        
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression algorithm using stochastic gradient descent.

    Parameters:
        y : numpy array of shape=(N, ).
        tx : numpy array of shape=(N, D+1).
        initial_w : numpy array of shape=(D+1, ) corresponding to the initialization for the model parameters.
        max_iters : a scalar denoting the total number of iterations of gradient descent.
        gamma : a scalar denoting the stepsize.
        
    Returns:
        loss: final loss value (scalar).
        w: model parameters as numpy array of shape (D+1, ).
    """
    w = initial_w
    N = len(y)
    
    for _ in range(max_iters):
        i = np.random.randint(0, N)
        gradient = compute_gradient_mse(np.array([y[i]]), np.array([tx[i]]), w)
        w = w - gamma * gradient
        
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations.

    Args:
        y : numpy array of shape (N, ).
        tx : numpy array of shape (N, D+1).
        
    Returns:
        loss: final loss value (scalar).
        w: model parameters as numpy array of shape (D+1, ).
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    Args:
        y : numpy array of shape=(N, ).
        tx : numpy array of shape=(N, D+1).
        lambda_ : a scalar denoting the regularization (penalty) term.
        
    Returns:
        loss: final loss value (scalar).
        w: model parameters as numpy array of shape (D+1, ).
    """
    w = np.linalg.solve(tx.T @ tx + lambda_* 2 * y.shape[0] * np.identity(tx.shape[1]), tx.T @ y)
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
        loss: final loss value (scalar).
        w: model parameters as numpy array of shape (D+1, ).
    """
    w = initial_w
    
    for _ in range(max_iters):
        gradient = compute_gradient_mle(y, tx, w)
        w = w - gamma * gradient

    loss = compute_loss_mle(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D+1)
        lambda_ : float, regularization parameter (penalty term)
        initial_w: numpy array of shape=(D+1, ). The initialization for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize.
        
    Returns:
        loss: final loss value (scalar).
        w: model parameters as numpy array of shape (D+1, ).
    """
    w = initial_w
    
    for _ in range(max_iters):
        gradient = compute_gradient_mle(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    
    loss = compute_loss_mle(y, tx, w)
    return w, loss


# -----------------------------------------INTERMEDIATE FUNCTIONS----------------------------------------------------#


def compute_loss_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        loss: the value of the loss (a scalar), corresponding to the input parameters w.
    """
    error = y - tx @ w
    loss = (1 / (2 * y.shape[0])) * error.T @ error
    return loss


def compute_gradient_mse(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D+1)
        w: numpy array of shape=(D+1, ). The vector of model parameters.

    Returns:
        gradient : numpy array of shape (D+1, ) (same shape as w), containing the gradient of the loss at w.
    """
    error = y - tx @ w
    new_w = (-1 / y.shape[0]) * tx.T @ error
    return new_w


def sigmoid(t):
    """Apply sigmoid function on t."""
    t = np.clip(t, -500, 500)  # np.exp(-500) is approximately 1e-308, to avoid overflow
    return 1 / (1 + np.exp(-t))


def compute_loss_mle(y, tx, w):
    """Compute the negative log-likelihood loss for logistic regression (MLE).

    Args:
        y : numpy array of shape (N, ). Actual labels (0 or 1).
        tx : numpy array of shape (N, D+1). Input features with bias term.
        w : numpy array of shape (D+1, ). Weight vector.

    Returns:
        loss : The mean negative log-likelihood loss.
    """
    y_pred = tx @ w
    #y_pred_prob = sigmoid(-y_pred)
    # loss = -np.mean(y * np.log(y_pred_prob) + (1 - y) * np.log(1 - y_pred_prob))
    t = np.clip(y_pred, -500, 500)
    loss = np.mean(np.log(1 + np.exp(t)) - y * y_pred)
    return loss


def compute_gradient_mle(y, tx, w):
    """Compute the gradient of the negative log-likelihood loss for logistic regression.

    Args:
        y : numpy array of shape=(N,)
        tx : numpy array of shape=(N, D+1)
        w : numpy array of shape=(D+1, )

    Returns:
        gradient : numpy array of shape (D+1, ) (same shape as w), containing the gradient of the loss at w.
    """
    pred = sigmoid(tx @ w)
    gradient = tx.T @ (pred - y) / y.shape[0]
    return gradient

def cross_validation_ridge_regression(yb, tx, lambdas):
    """
    Cross-validate ridge regression with different lambda values.

    Args:
        yb (np.ndarray): Labels of the data.
        tx (np.ndarray): Features of the data.
        lambdas (list): List of regularization parameters to test.

    Returns:
        dict: Dictionary containing the best lambda, F1 score, and weights.
    """
    best_lambda = None
    best_f1_score = -1
    best_w = None

    # Split data for training and testing (you may want to do k-fold cross-validation instead)
    tx_train, tx_test, y_train, y_test = split_data(tx, yb, 0.8, seed=1)

    for lambda_ in lambdas:
        # Train the model
        w, _ = ridge_regression(y_train, tx_train, lambda_)

        # Predict on the test set and calculate F1 score
        y_pred = np.where(tx_test @ w >= 0.5, 1, -1)
        f1 = f1_score(y_test, y_pred)

        # Update best lambda if this F1 score is higher
        if f1 > best_f1_score:
            best_lambda = lambda_
            best_f1_score = f1
            best_w = w

    # Return best lambda and associated metrics
    return {
        "lambda": best_lambda,
        "f1_score": best_f1_score,
        "weights": best_w,
    }