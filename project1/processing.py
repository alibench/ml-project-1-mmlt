import numpy as np
import matplotlib.pyplot as plt

def remove_nan(data, threshold):
    """
    Remove columns with too many NaN values, with respect to a threshold.

    Args:
        data (numpy.ndarray): Input data matrix.
        threshold (float): defined threshold for NaN values.

    Returns:
        numpy.ndarray: The cleaned input data matrix after removing columns with too many NaN values.
        tuple: A tuple containing the indices of the kept features.
    """
    nan_counts = np.isnan(data).sum(axis=0)
    max_nan_threshold = threshold * data.shape[0]
    columns_to_keep = nan_counts <= max_nan_threshold # only columns below threshold
    clean_data = data[:, columns_to_keep] # Remove columns with too many NaN values
    return clean_data, columns_to_keep

 
def compute_corr(data):
    """
    Compute the correlation matrix of the input data with pairwise deletion of missing values.

    Args:
        data (numpy.ndarray): Input data matrix.

    Returns:
        numpy.ndarray: Pairwise correlation matrix of the input data.
    """
    num_features = data.shape[1]
    corr_matrix = np.empty((num_features, num_features))
    
    for i in range(num_features):
        for j in range(num_features):
            valid_rows = ~np.isnan(data[:, i]) & ~np.isnan(data[:, j])
            corr_matrix[i, j] = np.corrcoef(data[valid_rows, i], data[valid_rows, j])[0, 1]
    
    return corr_matrix

def plot_corr_matrix(corr_matrix):
    """
    Plot the correlation matrix as a heatmap.

    Args:
        corr_matrix (numpy.ndarray): Input correlation matrix.
    """
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title('Correlation matrix')
    plt.show()