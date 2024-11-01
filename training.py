
import numpy as np
from implementations import *
from utils import *
from metrics import *


def evaluate_linear_regression_gd(
    y_train, 
    x_train,
    gamma_values,
    max_iters_values, 
    x_train_unbalanced, 
    y_train_unbalanced, 
    x_test, 
    degree_values=[1],  
    validation_split=0.3, 
    threshold=0
):
    """
    Performs grid search with a single train-validation split for logistic regression,
    integrating Precision, Recall, F1-score, and Accuracy calculation using NumPy only.
    Includes polynomial feature expansion.

    Parameters:
    - y_train: numpy array of training labels (-1 or 1)
    - x_train: numpy array of training features
    - gamma_values: list of gamma values to try
    - max_iters_values: list of maximum iterations values to try
    - x_train_unbalanced: numpy array of features of unbalanced training dataset
    - y_train_unbalanced: numpy array of labels of unbalanced training dataset
    - x_test: numpy array of test features
    - degree_values: list of integers, degrees for polynomial feature expansion (default [1])
    - validation_split: float, proportion of the dataset to include in the validation split (default 0.3)
    - threshold: float, decision threshold for classification (default 0.5)

    Returns:
    - best_params_logistic: dictionary containing the best 'gamma', 'max_iters', and 'degree'
    - best_f1_score_logistic: the best F1-score achieved on the validation set
    - best_accuracy_logistic: the corresponding accuracy on the validation set
    - y_test_pred: numpy array of predicted labels for the test set
    """
    
    # Initialize variables to store the best results
    best_params_lr = {}
    best_f1_score_lr = 0
    best_accuracy_lr = 0
    w_best = None
    
    # Shuffle data
    num_samples = y_train.shape[0]
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]
    
    # Split data into Training and Validation sets
    split_index = int(num_samples * (1 - validation_split))
    x_train_split = x_shuffled[:split_index]
    y_train_split = y_shuffled[:split_index]
    x_valid_split = x_shuffled[split_index:]
    y_valid_split = y_shuffled[split_index:]
    
    print("Starting grid search for Linear Regression (w/ gd) with feature expansion...")
    for degree in degree_values:
        print(f"\nFeature Expansion Degree = {degree}")
        
        # Apply polynomial feature expansion
        x_train_split_poly = poly_features(x_train_split, degree)
        x_valid_split_poly = poly_features(x_valid_split, degree)
        x_train_final_poly = poly_features(x_train_unbalanced, degree)
        
        for gamma in gamma_values:
            print(f"  For gamma = {gamma}")
            for max_iters in max_iters_values:
                print(f"    For max_iters = {max_iters}")
                
                # Initialize weights
                initial_w = np.zeros(x_train_split_poly.shape[1])
                
                # Train logistic regression on the training split
                w, _ = mean_squared_error_gd(y_train_split, x_train_split_poly, initial_w, max_iters, gamma)
                
                # Predict on validation set
                pred = np.dot(x_valid_split_poly, w)
                y_pred = np.where(pred >= threshold, 1, -1)
                
                # Calculate metrics on Validation Set
                accuracy = np.mean(y_pred == y_valid_split)
                precision = calculate_precision(y_valid_split, y_pred)
                recall = calculate_recall(y_valid_split, y_pred)
                f1 = calculate_f1_score(y_valid_split, y_pred)
                
                print(f"        Validation Metrics:")
                print(f"            Accuracy : {accuracy:.4f}")
                print(f"            Precision: {precision:.4f}")
                print(f"            Recall   : {recall:.4f}")
                print(f"            F1-score : {f1:.4f}")
                
                # Update best scores and parameters based on F1-score
                if f1 > best_f1_score_lr:
                    w_best = w
                    best_f1_score_lr = f1
                    best_accuracy_lr = accuracy  # Corresponding accuracy
                    best_params_lr = {
                        'gamma': gamma,
                        'max_iters': max_iters,
                        'degree': degree
                    }
                
                # Evaluate on the Original Training Set (x_train_final)
                pred_train = np.dot(x_train_final_poly, w)
                y_train_pred = np.where(pred_train >= threshold, 1, -1)
                train_accuracy = np.mean(y_train_pred == y_train_unbalanced)
                train_precision = calculate_precision(y_train_unbalanced, y_train_pred)
                train_recall = calculate_recall(y_train_unbalanced, y_train_pred)
                train_f1 = calculate_f1_score(y_train_unbalanced, y_train_pred)
                
                print(f"        Original Training Set Metrics:")
                print(f"            Accuracy : {train_accuracy:.4f}")
                print(f"            Precision: {train_precision:.4f}")
                print(f"            Recall   : {train_recall:.4f}")
                print(f"            F1-score : {train_f1:.4f}")
                print("-" * 60)
    
    # Apply polynomial feature expansion to final datasets using best degree
    best_degree = best_params_lr.get('degree', 1)
    x_test_final_poly_best = poly_features(x_test, best_degree)
    
    # Predict on the Test Set using the best weights
    pred_test = np.dot(x_test_final_poly_best, w_best)
    y_test_pred = np.where(pred_test >= threshold, 1, -1)
    
    return y_test_pred

def evaluate_linear_regression_sgd(
    y_train, 
    x_train,
    gamma_values,
    max_iters_values, 
    x_train_unbalanced, 
    y_train_unbalanced, 
    x_test, 
    degree_values=[1],  
    validation_split=0.3, 
    threshold=0
):
    """
    Performs grid search with a single train-validation split for logistic regression,
    integrating Precision, Recall, F1-score, and Accuracy calculation using NumPy only.
    Includes polynomial feature expansion.

    Parameters:
    - y_train: numpy array of training labels (-1 or 1)
    - x_train: numpy array of training features
    - gamma_values: list of gamma values to try
    - max_iters_values: list of maximum iterations values to try
    - x_train_unbalanced: numpy array of features of unbalanced training dataset
    - y_train_unbalanced: numpy array of labels of unbalanced training dataset
    - x_test: numpy array of test features
    - degree_values: list of integers, degrees for polynomial feature expansion (default [1])
    - validation_split: float, proportion of the dataset to include in the validation split (default 0.3)
    - threshold: float, decision threshold for classification (default 0.5)

    Returns:
    - best_params_logistic: dictionary containing the best 'gamma', 'max_iters', and 'degree'
    - best_f1_score_logistic: the best F1-score achieved on the validation set
    - best_accuracy_logistic: the corresponding accuracy on the validation set
    - y_test_pred: numpy array of predicted labels for the test set
    """
    
    # Initialize variables to store the best results
    best_params_lr = {}
    best_f1_score_lr = 0
    best_accuracy_lr = 0
    w_best = None
    
    # Shuffle data
    num_samples = y_train.shape[0]
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]
    
    # Split data into Training and Validation sets
    split_index = int(num_samples * (1 - validation_split))
    x_train_split = x_shuffled[:split_index]
    y_train_split = y_shuffled[:split_index]
    x_valid_split = x_shuffled[split_index:]
    y_valid_split = y_shuffled[split_index:]
    
    print("Starting grid search for Linear Regression (w/ sgd) with feature expansion...")
    for degree in degree_values:
        print(f"\nFeature Expansion Degree = {degree}")
        
        # Apply polynomial feature expansion
        x_train_split_poly = poly_features(x_train_split, degree)
        x_valid_split_poly = poly_features(x_valid_split, degree)
        x_train_final_poly = poly_features(x_train_unbalanced, degree)
        
        for gamma in gamma_values:
            print(f"  For gamma = {gamma}")
            for max_iters in max_iters_values:
                print(f"    For max_iters = {max_iters}")
                
                # Initialize weights
                initial_w = np.zeros(x_train_split_poly.shape[1])
                
                # Train logistic regression on the training split
                w, _ = mean_squared_error_sgd(y_train_split, x_train_split_poly, initial_w, max_iters, gamma)
                
                # Predict on validation set
                pred = np.dot(x_valid_split_poly, w)
                y_pred = np.where(pred >= threshold, 1, -1)
                
                # Calculate metrics on Validation Set
                accuracy = np.mean(y_pred == y_valid_split)
                precision = calculate_precision(y_valid_split, y_pred)
                recall = calculate_recall(y_valid_split, y_pred)
                f1 = calculate_f1_score(y_valid_split, y_pred)
                
                print(f"        Validation Metrics:")
                print(f"            Accuracy : {accuracy:.4f}")
                print(f"            Precision: {precision:.4f}")
                print(f"            Recall   : {recall:.4f}")
                print(f"            F1-score : {f1:.4f}")
                
                # Update best scores and parameters based on F1-score
                if f1 > best_f1_score_lr:
                    w_best = w
                    best_f1_score_lr = f1
                    best_accuracy_lr = accuracy  # Corresponding accuracy
                    best_params_lr = {
                        'gamma': gamma,
                        'max_iters': max_iters,
                        'degree': degree
                    }
                
                # Evaluate on the Original Training Set (x_train_final)
                pred_train = np.dot(x_train_final_poly, w)
                y_train_pred = np.where(pred_train >= threshold, 1, -1)
                train_accuracy = np.mean(y_train_pred == y_train_unbalanced)
                train_precision = calculate_precision(y_train_unbalanced, y_train_pred)
                train_recall = calculate_recall(y_train_unbalanced, y_train_pred)
                train_f1 = calculate_f1_score(y_train_unbalanced, y_train_pred)
                
                print(f"        Original Training Set Metrics:")
                print(f"            Accuracy : {train_accuracy:.4f}")
                print(f"            Precision: {train_precision:.4f}")
                print(f"            Recall   : {train_recall:.4f}")
                print(f"            F1-score : {train_f1:.4f}")
                print("-" * 60)
    
    # Apply polynomial feature expansion to final datasets using best degree
    best_degree = best_params_lr.get('degree', 1)
    x_test_final_poly_best = poly_features(x_test, best_degree)
    
    # Predict on the Test Set using the best weights
    pred_test = np.dot(x_test_final_poly_best, w_best)
    y_test_pred = np.where(pred_test >= threshold, 1, -1)
    
    return y_test_pred

def evaluate_least_squares(
    y_train, 
    x_train, 
    x_train_unbalanced, 
    y_train_unbalanced, 
    x_test_final, 
    degree=1, 
    validation_split=0.3, 
    threshold=0
):
    """
    Evaluates Least Squares Regression with polynomial feature expansion,
    integrating Precision, Recall, F1-score, and Accuracy calculation.
    Includes a train-validation split.

    Parameters:
    - y_train: numpy array of training labels (-1 or 1)
    - x_train: numpy array of training features
    - x_train_unbalanced: numpy array of features of unbalanced training dataset
    - y_train_unbalanced: numpy array of labels of unbalanced training dataset
    - x_test: numpy array of test features
    - degree: integer, degree for polynomial feature expansion (default 1)
    - validation_split: float, proportion of the dataset to include in the validation split (default 0.3)
    - threshold: float, decision threshold for classification (default 0)

    Returns:
    - y_test_pred: numpy array of predicted labels for the test set
    """
    
    # Shuffle data
    num_samples = y_train.shape[0]
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]
    
    # Split data into Training and Validation sets
    split_index = int(num_samples * (1 - validation_split))
    x_train_split = x_shuffled[:split_index]
    y_train_split = y_shuffled[:split_index]
    x_valid_split = x_shuffled[split_index:]
    y_valid_split = y_shuffled[split_index:]
    
    print("Starting evaluation for Least Squares Regression with feature expansion...")
    
    # Apply polynomial feature expansion to training, validation and original unbalanced dataset
    x_train_split_poly = poly_features(x_train_split, degree)
    x_valid_split_poly = poly_features(x_valid_split, degree)
    x_train_unbalanced_poly = poly_features(x_train_unbalanced, degree)
    
    # Train Least Squares Regression on the training split
    w, loss = least_squares(y_train_split, x_train_split_poly)
    
    # Predict on validation set
    y_pred_continuous = np.dot(x_valid_split_poly, w)
    y_pred = np.where(y_pred_continuous >= threshold, 1, -1)
    
    # Calculate metrics on Validation Set
    accuracy = np.mean(y_pred == y_valid_split)
    precision = calculate_precision(y_valid_split, y_pred)
    recall = calculate_recall(y_valid_split, y_pred)
    f1 = calculate_f1_score(y_valid_split, y_pred)
    
    print(f"    Loss on training (w/ cross validation): {loss:.4f}")
    print(f"Validation Metrics:")
    print(f"    Accuracy : {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall   : {recall:.4f}")
    print(f"    F1-score : {f1:.4f}")
    
    # Evaluate on the Original Training Set (x_train_final)
    y_train_pred_continuous = np.dot(x_train_unbalanced_poly, w)
    y_train_pred = np.where(y_train_pred_continuous >= threshold, 1, -1)
    train_accuracy = np.mean(y_train_pred == y_train_unbalanced)
    train_precision = calculate_precision(y_train_unbalanced, y_train_pred)
    train_recall = calculate_recall(y_train_unbalanced, y_train_pred)
    train_f1 = calculate_f1_score(y_train_unbalanced, y_train_pred)
    
    print(f"Training Set Metrics:")
    print(f"    Accuracy : {train_accuracy:.4f}")
    print(f"    Precision: {train_precision:.4f}")
    print(f"    Recall   : {train_recall:.4f}")
    print(f"    F1-score : {train_f1:.4f}")
    print("-" * 80)
    
    # Apply polynomial feature expansion to final test dataset
    x_test_final_poly = poly_features(x_test_final, degree)
    
    # Predict on the Test Set using the weights
    y_test_pred_continuous = np.dot(x_test_final_poly, w)
    y_test_pred = np.where(y_test_pred_continuous >= threshold, 1, -1)
    
    return y_test_pred

def evaluate_ridge_regression(
    y_train, 
    x_train, 
    lambda_values,
    x_train_unbalanced, 
    y_train_unbalanced, 
    x_test, 
    degree_values=[1], 
    validation_split=0.3, 
    threshold=0
):
    """
    Performs grid search with a single train-validation split for Ridge Regression,
    integrating Precision, Recall, F1-score, and Accuracy calculation using NumPy only.
    Includes polynomial feature expansion.

    Parameters:
    - y_train: numpy array of training labels (-1 or 1)
    - x_train: numpy array of training features
    - lambda_values: list of lambda values to try for regularization
    - x_train_unbalanced: numpy array of features of unbalanced training dataset
    - y_train_unbalanced: numpy array of labels of unbalanced training dataset
    - x_test: numpy array of test features
    - degree_values: list of integers, degrees for polynomial feature expansion (default [1])
    - validation_split: float, proportion of the dataset to include in the validation split (default 0.3)
    - threshold: float, decision threshold for classification (default 0)

    Returns:
    - best_params_ridge: dictionary containing the best 'lambda_' and 'degree'
    - best_f1_score_ridge: the best F1-score achieved on the validation set
    - best_accuracy_ridge: the corresponding accuracy on the validation set
    - y_test_pred: numpy array of predicted labels for the test set
    """
    
    # Initialize variables to store the best results
    best_params_ridge = {}
    best_f1_score_ridge = -np.inf 
    best_accuracy_ridge = 0
    w_best = None
    
    # Shuffle data
    num_samples = y_train.shape[0]
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]
    
    # Split data into Training and Validation sets
    split_index = int(num_samples * (1 - validation_split))
    x_train_split = x_shuffled[:split_index]
    y_train_split = y_shuffled[:split_index]
    x_valid_split = x_shuffled[split_index:]
    y_valid_split = y_shuffled[split_index:]
    
    print("Starting grid search for Ridge Regression with feature expansion...")
    for degree in degree_values:
        print(f"\nFeature Expansion Degree = {degree}")
        
        # Apply polynomial feature expansion
        x_train_split_poly = poly_features(x_train_split, degree)
        x_valid_split_poly = poly_features(x_valid_split, degree)
        x_train_final_poly = poly_features(x_train_unbalanced, degree)
        
        for lambda_ in lambda_values:
            print(f"  For lambda = {lambda_}")
            
            # Train Ridge Regression on the training split
            w, loss = ridge_regression(y_train_split, x_train_split_poly, lambda_)
            
            # Predict on validation set
            y_pred_continuous = np.dot(x_valid_split_poly, w)
            y_pred = np.where(y_pred_continuous >= threshold, 1, -1)
            
            # Calculate metrics on Validation Set
            accuracy = np.mean(y_pred == y_valid_split)
            precision = calculate_precision(y_valid_split, y_pred)
            recall = calculate_recall(y_valid_split, y_pred)
            f1 = calculate_f1_score(y_valid_split, y_pred)
            
            print(f"    Loss on training (w/ cross validation): {loss:.4f}")
            print(f"      Validation Metrics:")
            print(f"          Accuracy : {accuracy:.4f}")
            print(f"          Precision: {precision:.4f}")
            print(f"          Recall   : {recall:.4f}")
            print(f"          F1-score : {f1:.4f}")
            
            # Update best scores and parameters based on F1-score
            if f1 > best_f1_score_ridge:
                w_best = w
                best_f1_score_ridge = f1
                best_accuracy_ridge = accuracy  # Corresponding accuracy
                best_params_ridge = {
                    'lambda_': lambda_,
                    'degree': degree
                }
            
            # Evaluate on the Original Training Set (x_train_final)
            y_train_pred_continuous = np.dot(x_train_final_poly, w)
            y_train_pred = np.where(y_train_pred_continuous >= threshold, 1, -1)
            train_accuracy = np.mean(y_train_pred == y_train_unbalanced)
            train_precision = calculate_precision(y_train_unbalanced, y_train_pred)
            train_recall = calculate_recall(y_train_unbalanced, y_train_pred)
            train_f1 = calculate_f1_score(y_train_unbalanced, y_train_pred)
            
            print(f"      Original Training Set Metrics:")
            print(f"          Accuracy : {train_accuracy:.4f}")
            print(f"          Precision: {train_precision:.4f}")
            print(f"          Recall   : {train_recall:.4f}")
            print(f"          F1-score : {train_f1:.4f}")
            print("-" * 80)
    
    # Apply polynomial feature expansion to final datasets using best degree
    best_degree = best_params_ridge.get('degree', 1)
    x_test_final_poly_best = poly_features(x_test, best_degree)
    
    # Predict on the Test Set using the best weights
    y_test_pred_continuous = np.dot(x_test_final_poly_best, w_best)
    y_test_pred = np.where(y_test_pred_continuous >= threshold, 1, -1)
    
    return y_test_pred

def evaluate_logistic_regression(
    y_train, 
    x_train, 
    gamma_values, 
    max_iters_values, 
    x_train_final, 
    y_train_final, 
    x_test_final, 
    degree_values=[1],  
    validation_split=0.3, 
    threshold=0.5
):
    """
    Performs grid search with a single train-validation split for logistic regression,
    integrating Precision, Recall, F1-score, and Accuracy calculation using NumPy only.
    Includes polynomial feature expansion.

    Parameters:
    - y_train: numpy array of training labels (-1 or 1)
    - x_train: numpy array of training features
    - gamma_values: list of gamma values to try
    - max_iters_values: list of maximum iterations values to try
    - x_train_final: numpy array of features for final training evaluation
    - y_train_final: numpy array of labels for final training evaluation
    - x_test_final: numpy array of test features
    - degree_values: list of integers, degrees for polynomial feature expansion (default [1])
    - validation_split: float, proportion of the dataset to include in the validation split (default 0.3)
    - threshold: float, decision threshold for classification (default 0.5)

    Returns:
    - best_params_logistic: dictionary containing the best 'gamma', 'max_iters', and 'degree'
    - best_f1_score_logistic: the best F1-score achieved on the validation set
    - best_accuracy_logistic: the corresponding accuracy on the validation set
    - y_test_pred: numpy array of predicted labels for the test set
    """
    
    # Convert labels from -1 to 0 for binary classification
    y_train_binary = np.where(y_train == -1, 0, y_train)
    y_train_final_binary = np.where(y_train_final == -1, 0, y_train_final)
    
    # Initialize variables to store the best results
    best_params_logistic = {}
    best_f1_score_logistic = -np.inf
    best_accuracy_logistic = 0
    w_best = None
    
    # Shuffle data
    num_samples = y_train_binary.shape[0]
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    x_shuffled = x_train[indices]
    y_shuffled = y_train_binary[indices]
    
    # Split data into Training and Validation sets
    split_index = int(num_samples * (1 - validation_split))
    x_train_split = x_shuffled[:split_index]
    y_train_split = y_shuffled[:split_index]
    x_valid_split = x_shuffled[split_index:]
    y_valid_split = y_shuffled[split_index:]
    
    print("Starting grid search for Logistic Regression with feature expansion...")
    for degree in degree_values:
        print(f"\nFeature Expansion Degree = {degree}")
        
        # Apply polynomial feature expansion
        x_train_split_poly = poly_features(x_train_split, degree)
        x_valid_split_poly = poly_features(x_valid_split, degree)
        x_train_final_poly = poly_features(x_train_final, degree)
        
        for gamma in gamma_values:
            print(f"  For gamma = {gamma}")
            for max_iters in max_iters_values:
                print(f"    For max_iters = {max_iters}")
                
                # Initialize weights
                initial_w = np.zeros(x_train_split_poly.shape[1])
                
                # Train logistic regression on the training split
                w, loss = logistic_regression(y_train_split, x_train_split_poly, initial_w, max_iters, gamma)
                
                # Predict on validation set
                pred_probs = sigmoid(np.dot(x_valid_split_poly, w))
                y_pred = (pred_probs >= threshold).astype(int)
                
                # Calculate metrics on Validation Set
                accuracy = np.mean(y_pred == y_valid_split)
                precision = calculate_precision(y_valid_split, y_pred)
                recall = calculate_recall(y_valid_split, y_pred)
                f1 = calculate_f1_score(y_valid_split, y_pred)
                
                print(f"    Loss on training (w/ cross validation): {loss:.4f}")
                print(f"        Validation Metrics:")
                print(f"            Accuracy : {accuracy:.4f}")
                print(f"            Precision: {precision:.4f}")
                print(f"            Recall   : {recall:.4f}")
                print(f"            F1-score : {f1:.4f}")
                
                # Update best scores and parameters based on F1-score
                if f1 > best_f1_score_logistic:
                    w_best = w
                    best_f1_score_logistic = f1
                    best_accuracy_logistic = accuracy  # Corresponding accuracy
                    best_params_logistic = {
                        'gamma': gamma,
                        'max_iters': max_iters,
                        'degree': degree
                    }
                
                # Evaluate on the Original Training Set (x_train_final)
                pred_train_probs = sigmoid(np.dot(x_train_final_poly, w))
                y_train_pred = (pred_train_probs >= threshold).astype(int)
                train_accuracy = np.mean(y_train_pred == y_train_final_binary)
                train_precision = calculate_precision(y_train_final_binary, y_train_pred)
                train_recall = calculate_recall(y_train_final_binary, y_train_pred)
                train_f1 = calculate_f1_score(y_train_final_binary, y_train_pred)
                
                print(f"        Original Training Set Metrics:")
                print(f"            Accuracy : {train_accuracy:.4f}")
                print(f"            Precision: {train_precision:.4f}")
                print(f"            Recall   : {train_recall:.4f}")
                print(f"            F1-score : {train_f1:.4f}")
                print("-" * 60)
    
    # Apply polynomial feature expansion to final datasets using best degree
    best_degree = best_params_logistic.get('degree', 1)
    x_test_final_poly_best = poly_features(x_test_final, best_degree)
    
    # Predict on the Test Set using the best weights
    pred_test_probs = sigmoid(np.dot(x_test_final_poly_best, w_best))
    y_test_pred = (pred_test_probs >= threshold).astype(int)
    
    return y_test_pred

def evaluate_reg_logistic_regression(
    y_train, 
    x_train, 
    gamma_values, 
    max_iters_values, 
    x_train_final, 
    y_train_final, 
    x_test_final, 
    lambda_values,
    degree_values=[1], 
    validation_split=0.3, 
    threshold=0.5
):
    """
    Performs grid search with a single train-validation split for regularized logistic regression,
    integrating Precision, Recall, F1-score, and Accuracy calculation using NumPy only.
    Includes polynomial feature expansion.

    Parameters:
    - y_train: numpy array of training labels (-1 or 1)
    - x_train: numpy array of training features
    - gamma_values: list of gamma values to try
    - max_iters_values: list of maximum iterations values to try
    - x_train_final: numpy array of features for final training evaluation
    - y_train_final: numpy array of labels for final training evaluation
    - x_test_final: numpy array of test features
    - lambda_values: list of lambda values to try for regularization
    - degree_values: list of integers, degrees for polynomial feature expansion (default [1])
    - validation_split: float, proportion of the dataset to include in the validation split (default 0.3)
    - threshold: float, decision threshold for classification (default 0.5)

    Returns:
    - best_params_logistic: dictionary containing the best 'gamma', 'max_iters', 'lambda_', and 'degree'
    - best_f1_score_logistic: the best F1-score achieved on the validation set
    - best_accuracy_logistic: the corresponding accuracy on the validation set
    - y_test_pred: numpy array of predicted labels for the test set
    """
    
    # Convert labels from -1 to 0 for binary classification
    y_train_binary = np.where(y_train == -1, 0, y_train)
    y_train_final_binary = np.where(y_train_final == -1, 0, y_train_final)
    
    # Initialize variables to store the best results
    best_params_logistic = {}
    best_f1_score_logistic = -np.inf
    best_accuracy_logistic = 0
    w_best = None
    
    # Shuffle data
    num_samples = y_train_binary.shape[0]
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    x_shuffled = x_train[indices]
    y_shuffled = y_train_binary[indices]
    
    # Split data into Training and Validation sets
    split_index = int(num_samples * (1 - validation_split))
    x_train_split = x_shuffled[:split_index]
    y_train_split = y_shuffled[:split_index]
    x_valid_split = x_shuffled[split_index:]
    y_valid_split = y_shuffled[split_index:]
    
    print("Starting grid search for Regularized Logistic Regression with feature expansion...")
    for degree in degree_values:
        print(f"\nFeature Expansion Degree = {degree}")
        
        # Apply polynomial feature expansion
        x_train_split_poly = poly_features(x_train_split, degree)
        x_valid_split_poly = poly_features(x_valid_split, degree)
        x_train_final_poly = poly_features(x_train_final, degree)
        
        for lambda_ in lambda_values:
            print(f"  For lambda = {lambda_}")
            for gamma in gamma_values:
                print(f"    For gamma = {gamma}")
                for max_iters in max_iters_values:
                    print(f"      For max_iters = {max_iters}")
                    
                    # Initialize weights
                    initial_w = np.zeros(x_train_split_poly.shape[1])
                    
                    # Train regularized logistic regression on the training split
                    w, loss = reg_logistic_regression(y_train_split, x_train_split_poly, lambda_, initial_w, max_iters, gamma)
                    
                    # Predict on validation set
                    pred_probs = sigmoid(np.dot(x_valid_split_poly, w))
                    y_pred = (pred_probs >= threshold).astype(int)
                    
                    # Calculate metrics on Validation Set
                    accuracy = np.mean(y_pred == y_valid_split)
                    precision = calculate_precision(y_valid_split, y_pred)
                    recall = calculate_recall(y_valid_split, y_pred)
                    f1 = calculate_f1_score(y_valid_split, y_pred)
                    print(f"    Loss on training (w/ cross validation): {loss:.4f}")
                    print(f"          Validation Metrics:")
                    print(f"              Accuracy : {accuracy:.4f}")
                    print(f"              Precision: {precision:.4f}")
                    print(f"              Recall   : {recall:.4f}")
                    print(f"              F1-score : {f1:.4f}")
                    
                    # Update best scores and parameters based on F1-score
                    if f1 > best_f1_score_logistic:
                        w_best = w
                        best_f1_score_logistic = f1
                        best_accuracy_logistic = accuracy  # Corresponding accuracy
                        best_params_logistic = {
                            'lambda_': lambda_,
                            'gamma': gamma,
                            'max_iters': max_iters,
                            'degree': degree
                        }
                    
                    # Evaluate on the Original Training Set (x_train_final)
                    pred_train_probs = sigmoid(np.dot(x_train_final_poly, w))
                    y_train_pred = (pred_train_probs >= threshold).astype(int)
                    train_accuracy = np.mean(y_train_pred == y_train_final_binary)
                    train_precision = calculate_precision(y_train_final_binary, y_train_pred)
                    train_recall = calculate_recall(y_train_final_binary, y_train_pred)
                    train_f1 = calculate_f1_score(y_train_final_binary, y_train_pred)
                    
                    print(f"          Original Training Set Metrics:")
                    print(f"              Accuracy : {train_accuracy:.4f}")
                    print(f"              Precision: {train_precision:.4f}")
                    print(f"              Recall   : {train_recall:.4f}")
                    print(f"              F1-score : {train_f1:.4f}")
                    print("-" * 80)
    
    # Apply polynomial feature expansion to final datasets using best degree
    best_degree = best_params_logistic.get('degree', 1)
    x_test_final_poly_best = poly_features(x_test_final, best_degree)
    
    # Predict on the Test Set using the best weights
    pred_test_probs = sigmoid(np.dot(x_test_final_poly_best, w_best))
    y_test_pred = (pred_test_probs >= threshold).astype(int)
    
    return y_test_pred

































