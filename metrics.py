import numpy as np

def confusion_matrix(y_true, y_pred):
    """Compute the confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        
    Returns:
        numpy.ndarray: The confusion matrix.
    """
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    matrix = np.array([[TN, FP], [FN, TP]])
    return (matrix/np.sum(matrix))

def calculate_precision(y_true, y_pred):
    """
    Calculates the precision metric for binary classification.
    Supports labels with values -1 and 1, or 0 and 1.

    Parameters:
    - y_true: numpy array of true labels (-1 and 1, or 0 and 1)
    - y_pred: numpy array of predicted labels (-1 and 1, or 0 and 1)

    Returns:
    - precision: float
    """
    # Map labels to 0 and 1
    y_true_mapped = np.where(y_true == -1, 0, y_true)
    y_pred_mapped = np.where(y_pred == -1, 0, y_pred)
    
    true_positive = np.sum((y_pred_mapped == 1) & (y_true_mapped == 1))
    predicted_positive = np.sum(y_pred_mapped == 1)
    if predicted_positive == 0:
        return 0.0
    precision = true_positive / predicted_positive
    return precision

def calculate_recall(y_true, y_pred):
    """
    Calculates the recall metric for binary classification.
    Supports labels with values -1 and 1, or 0 and 1.

    Parameters:
    - y_true: numpy array of true labels (-1 and 1, or 0 and 1)
    - y_pred: numpy array of predicted labels (-1 and 1, or 0 and 1)

    Returns:
    - recall: float
    """
    # Map labels to 0 and 1
    y_true_mapped = np.where(y_true == -1, 0, y_true)
    y_pred_mapped = np.where(y_pred == -1, 0, y_pred)
    
    true_positive = np.sum((y_pred_mapped == 1) & (y_true_mapped == 1))
    actual_positive = np.sum(y_true_mapped == 1)
    if actual_positive == 0:
        return 0.0
    recall = true_positive / actual_positive
    return recall

def calculate_f1_score(y_true, y_pred):
    """
    Calculates the F1-score metric.

    Parameters:
    - y_true: numpy array of true labels (-1 and 1, or 0 and 1)
    - y_pred: numpy array of predicted labels (-1 and 1, or 0 and 1)

    Returns:
    - f1_score: float
    """
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    if (precision + recall) == 0:
        return 0.0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score