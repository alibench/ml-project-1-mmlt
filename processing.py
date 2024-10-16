import numpy as np
import matplotlib.pyplot as plt

def replace_placeholders_with_nan(x_train, x_test):
    """
    Replace specific placeholder values with NaN based on the following rules:
    
    For each column:
        - If the column contains 777777 or 999999, replace all 777777 and 999999 with NaN.
        - Else if the column contains 99900 or 99000, replace all 99900 and 99000 with NaN.
        - Else if the column contains 7777 or 9999, replace all 7777 and 9999 with NaN.
        - Else if the column contains 777 or 999, replace all 777 and 999 with NaN.
        - Else if the column contains 77 or 99, replace all 77 and 99 with NaN.
        - Else, replace all 7 and 9 with NaN.
    
    Args:
        x_train (np.array): Training data array.
        x_test (np.array): Test data array.
        
    Returns:
        x_train_clean (np.array): Cleaned training data.
        x_test_clean (np.array): Cleaned test data.
    """
    # Make copies to avoid modifying original data
    x_train_clean = x_train.copy()
    x_test_clean = x_test.copy()
    
    num_features = x_train_clean.shape[1]
    
    for col in range(num_features):
        # Extract the column data
        train_col = x_train_clean[:, col]
        
        # Check for the presence of 777777 or 999999
        has_777777 = np.isin([777777, 999999], train_col).any()
        if has_777777:
            # Replace 777777 and 9999 with NaN in both train and test
            x_train_clean[:, col] = np.where(np.isin(train_col, [777777, 999999]), np.nan, train_col)
            x_test_clean[:, col] = np.where(np.isin(x_test_clean[:, col], [777777, 999999]), np.nan, x_test_clean[:, col])
            continue  # Move to the next column
        
        # Check for the presence of 99900 or 99000
        has_99900 = np.isin([99900, 99000], train_col).any()
        if has_99900:
            # Replace 99900 and 99000 with NaN in both train and test
            x_train_clean[:, col] = np.where(np.isin(train_col, [99900, 99000]), np.nan, train_col)
            x_test_clean[:, col] = np.where(np.isin(x_test_clean[:, col], [99900, 99000]), np.nan, x_test_clean[:, col])
            continue  # Move to the next column
        
        # Check for the presence of 7777 or 9799
        has_7777 = np.isin([7777, 9999], train_col).any()
        if has_7777:
            # Replace 7777 and 9999 with NaN in both train and test
            x_train_clean[:, col] = np.where(np.isin(train_col, [7777, 9999]), np.nan, train_col)
            x_test_clean[:, col] = np.where(np.isin(x_test_clean[:, col], [7777, 9999]), np.nan, x_test_clean[:, col])
            continue  # Move to the next column
        
        # Check for the presence of 777 or 999
        has_777 = np.isin([777, 999], train_col).any()
        if has_777:
            # Replace 777 and 999 with NaN in both train and test
            x_train_clean[:, col] = np.where(np.isin(train_col, [777, 999]), np.nan, train_col)
            x_test_clean[:, col] = np.where(np.isin(x_test_clean[:, col], [777, 999]), np.nan, x_test_clean[:, col])
            continue  # Move to the next column
        
        # Else, check for the presence of 77 or 99
        has_77 = np.isin([77, 99], train_col).any()
        if has_77:
            # Replace 77 and 99 with NaN in both train and test
            x_train_clean[:, col] = np.where(np.isin(train_col, [77, 99]), np.nan, train_col)
            x_test_clean[:, col] = np.where(np.isin(x_test_clean[:, col], [77, 99]), np.nan, x_test_clean[:, col])
            continue  # Move to the next column
        
        # Else, replace 7 and 9 with NaN
        x_train_clean[:, col] = np.where(train_col == 7, np.nan, train_col)
        x_train_clean[:, col] = np.where(x_train_clean[:, col] == 9, np.nan, x_train_clean[:, col])
        
        x_test_clean[:, col] = np.where(x_test_clean[:, col] == 7, np.nan, x_test_clean[:, col])
        x_test_clean[:, col] = np.where(x_test_clean[:, col] == 9, np.nan, x_test_clean[:, col])
    
    return x_train_clean, x_test_clean


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


def classify_features(data, threshold):
    """
    Classify features as categorical or continuous based on the number of unique values.

    Args:
        data (numpy.ndarray): Input data matrix.
        threshold (int): Maximum number of unique values to consider a feature as categorical.

    Returns:
        tuple: Two lists containing the indices of categorical and continuous features.
    """
    # Initialize lists to hold feature indices
    categorical_features = []
    continuous_features = []

    # Loop through each feature (column) in the data
    for idx in range(data.shape[1]):
        unique_values = np.unique(data[:, idx][~np.isnan(data[:, idx])])  # Ignore NaNs when counting unique values
        unique_count = len(unique_values)

        # Classify based on the threshold
        if unique_count <= threshold:
            categorical_features.append(idx)
        else:
            continuous_features.append(idx)
    
    return categorical_features, continuous_features

def impute_median_for_continuous_features(x_train, x_test, continuous_indices):
    """
    Impute NaN values in continuous features with the median.

    Args:
        x_train (np.ndarray): Training data with shape (n_samples, n_features).
        x_test (np.ndarray): Test data with shape (n_samples, n_features).
        continuous_indices (list): List of indices corresponding to continuous features.

    Returns:
        np.ndarray: Imputed training data.
        np.ndarray: Imputed test data.
    """
    for idx in continuous_indices:
        # Impute training data
        col = x_train[:, idx]
        median = np.nanmedian(col)
        col = np.where(np.isnan(col), median, col)
        x_train[:, idx] = col

        # Impute test data
        test_col = x_test[:, idx]
        test_col = np.where(np.isnan(test_col), median, test_col)
        x_test[:, idx] = test_col

    return x_train, x_test

def impute_mode(x, feature_indices):
    for idx in feature_indices:
        col = x[:, idx]
        # Compute the mode, ignoring NaNs
        unique, counts = np.unique(col[~np.isnan(col)], return_counts=True)
        mode = unique[np.argmax(counts)]
        # Replace NaNs with mode
        col = np.where(np.isnan(col), mode, col)
        x[:, idx] = col
    return x

class CapOutliersResult:
    def __init__(self, x_train_capped, x_test_capped, lower_bounds, upper_bounds):
        self.x_train_capped = x_train_capped
        self.x_test_capped = x_test_capped
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

def cap_outliers(x_train, x_test, continuous_indices, lower_percentile=1, upper_percentile=99):
    """
    Caps outliers in continuous features based on specified percentiles.

    Args:
        x_train (np.ndarray): Training data with shape (n_samples, n_features).
        x_test (np.ndarray): Test data with shape (n_samples, n_features).
        continuous_indices (list): List of indices corresponding to continuous features.
        lower_percentile (float): Lower percentile for capping (default is 1).
        upper_percentile (float): Upper percentile for capping (default is 99).

    Returns:
        CapOutliersResult: A result object containing capped data and capping bounds.
    """
    # Copy the data to avoid modifying the original arrays
    x_train_capped = x_train.copy()
    x_test_capped = x_test.copy()
    
    lower_bounds = {}
    upper_bounds = {}
    
    for idx in continuous_indices:
        # Calculate the lower and upper percentile values from the training data
        lower_value = np.percentile(x_train[:, idx], lower_percentile)
        upper_value = np.percentile(x_train[:, idx], upper_percentile)
        
        # Store the calculated bounds
        lower_bounds[idx] = lower_value
        upper_bounds[idx] = upper_value
        
        # Cap the values in the training data
        x_train_capped[:, idx] = np.where(x_train[:, idx] < lower_value, lower_value,
                                          np.where(x_train[:, idx] > upper_value, upper_value, x_train[:, idx]))
        
        # Cap the values in the test data
        x_test_capped[:, idx] = np.where(x_test[:, idx] < lower_value, lower_value,
                                         np.where(x_test[:, idx] > upper_value, upper_value, x_test[:, idx]))
    
    return CapOutliersResult(x_train_capped, x_test_capped, lower_bounds, upper_bounds)


def verify_capping(x, continuous_indices, lower_bounds, upper_bounds):
    """
    Verifies that the capping of outliers worked as expected.

    Args:
        x (np.ndarray): Data to verify with shape (n_samples, n_features).
        continuous_indices (list): List of indices corresponding to continuous features.
        lower_bounds (dict): Dictionary of lower bounds used for capping.
        upper_bounds (dict): Dictionary of upper bounds used for capping.

    Returns:
        None
    """
    for idx in continuous_indices:
        col = x[:, idx]
        lower_bound = lower_bounds[idx]
        upper_bound = upper_bounds[idx]
        
        # Check if any value is below the lower bound
        below = np.sum(col < lower_bound)
        if below > 0:
            print(f"Feature {idx}: {below} values below the lower bound ({lower_bound}).")
        
        # Check if any value is above the upper bound
        above = np.sum(col > upper_bound)
        if above > 0:
            print(f"Feature {idx}: {above} values above the upper bound ({upper_bound}).")
    
    print("Verification completed.")
 
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

def remove_correlated_features(data, corr_matrix, threshold):
    """
    Remove correlated features from the input data based on the correlation matrix.

    Args:
        data (numpy.ndarray): Input data matrix.
        corr_matrix (numpy.ndarray): Pairwise correlation matrix of the input data.
        threshold (float): defined threshold for correlation values.

    Returns:
        numpy.ndarray: The cleaned input data matrix after removing correlated features.
        tuple: A tuple containing the indices of the kept features.
    """
    correlated_features = np.where(np.abs(corr_matrix) > threshold)
    features_to_keep = np.arange(data.shape[1])
    for i, j in zip(correlated_features[0], correlated_features[1]): 
        if i != j and i in features_to_keep and j in features_to_keep: # Remove one of the two correlated features
            features_to_keep = features_to_keep[features_to_keep != j] 
    filtered_data = data[:, features_to_keep]
    return filtered_data, features_to_keep

def encode_categorical_features_mixed(x, binary_indices, nominal_indices):
    """
    Encode categorical features using Label Encoding for binary features and One-Hot Encoding for nominal features.

    Args:
        x (np.ndarray): Input data array with shape (n_samples, n_features).
        binary_indices (list): List of column indices for binary categorical features.
        nominal_indices (list): List of column indices for nominal categorical features.

    Returns:
        np.ndarray: Data array with encoded categorical features.
    """
    # Label Encoding for Binary Features
    for idx in binary_indices:
        col = x[:, idx]
        unique = np.unique(col)
        if len(unique) != 2:
            print(f"Warning: Feature {idx} is not binary. Skipping Label Encoding.")
            continue
        mapping = {unique[0]: 0, unique[1]: 1}
        x[:, idx] = np.vectorize(mapping.get)(col)
    
    # One-Hot Encoding for Nominal Features
    one_hot_columns = []
    for idx in nominal_indices:
        col = x[:, idx].astype(int)
        unique_categories = np.unique(col)
        unique_categories = unique_categories[~np.isnan(unique_categories)]
        #print(f'One-Hot Encoding Feature {idx} with categories: {unique_categories}')
        for category in unique_categories:
            new_col = (col == category).astype(int).reshape(-1, 1)
            one_hot_columns.append(new_col)
    
    if one_hot_columns:
        one_hot_matrix = np.hstack(one_hot_columns)
        # Remove original nominal categorical columns
        x = np.delete(x, nominal_indices, axis=1)
        # Append One-Hot Encoded columns
        x = np.hstack((x, one_hot_matrix))
    
    return x

def standardize_features(x_train, x_test, continuous_indices):
    """
    Standardize continuous features to have zero mean and unit variance.

    Args:
        x_train (np.ndarray): Training data with shape (n_samples, n_features).
        x_test (np.ndarray): Test data with shape (n_samples, n_features).
        continuous_indices (list): List of column indices corresponding to continuous features.

    Returns:
        np.ndarray: Scaled training data.
        np.ndarray: Scaled test data.
    """
    # Calculate mean and std from training data
    mean = np.mean(x_train[:, continuous_indices], axis=0)
    std = np.std(x_train[:, continuous_indices], axis=0)
    
    # Prevent division by zero
    std_replaced = np.where(std == 0, 1, std)
    
    # Standardize training data
    x_train[:, continuous_indices] = (x_train[:, continuous_indices] - mean) / std_replaced
    
    # Standardize test data using training mean and std
    x_test[:, continuous_indices] = (x_test[:, continuous_indices] - mean) / std_replaced
    
    return x_train, x_test

def variance_threshold(train_data, test_data, threshold=0.0):
    """
    Remove features with variance below the specified threshold.

    Args:
        x (np.ndarray): Input data array.
        threshold (float): Variance threshold.

    Returns:
        np.ndarray: Data array with low-variance features removed.
    """
    variances = np.var(train_data, axis=0)
    features_to_keep = variances > threshold
    return train_data[:, features_to_keep], test_data[:, features_to_keep]


def smote(x, y, minority_class, k=5, num_samples=None):
    """
    Implement SMOTE to balance the dataset.

    Args:
        x (np.ndarray): Feature matrix.
        y (np.ndarray): Target array.
        minority_class (int or float): The class value that is the minority class.
        k (int): The number of nearest neighbors to consider.
        num_samples (int): Number of synthetic samples to generate. If None, generate enough to balance the classes.

    Returns:
        np.ndarray: New feature matrix with synthetic samples added.
        np.ndarray: New target array with synthetic labels added.
    """
    minority_indices = np.where(y == minority_class)[0]
    x_minority = x[minority_indices]

    # Determine the number of synthetic samples to generate
    if num_samples is None:
        majority_class_count = np.sum(y != minority_class)
        minority_class_count = len(minority_indices)
        num_samples = majority_class_count - minority_class_count
    print(f"Generating {num_samples} synthetic samples.")
    
    synthetic_samples = []
    
    # For each minority sample, find its k-nearest neighbors
    for i in range(num_samples):
        # Randomly select a minority sample
        idx = np.random.randint(0, x_minority.shape[0])
        sample = x_minority[idx]
        
        # Euclidean distances to all other minority samples
        distances = np.linalg.norm(x_minority - sample, axis=1)
        
        # Sort and select the k nearest neighbors
        nearest_indices = np.argsort(distances)[1:k+1]
        neighbor_idx = np.random.choice(nearest_indices)
        
        # Select a random neighbor
        neighbor = x_minority[neighbor_idx]
        
        # Generate a synthetic sample by interpolating between the sample and its neighbor
        diff = neighbor - sample
        gap = np.random.rand()  
        synthetic_sample = sample + gap * diff
        
        synthetic_samples.append(synthetic_sample)

    # Convert the synthetic samples list to a numpy array
    synthetic_samples = np.array(synthetic_samples)
    
    # Create synthetic labels
    synthetic_labels = np.full(synthetic_samples.shape[0], minority_class)

    # Combine original data with synthetic data
    x_new = np.vstack((x, synthetic_samples))
    y_new = np.hstack((y, synthetic_labels))

    return x_new, y_new
