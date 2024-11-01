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


def remove_correlated_features_with_catContUpdate(data, corr_matrix, threshold, categorical_features, continuous_features):
    """
    Remove correlated features from the input data based on the correlation matrix.
    
    Args:
        data (numpy.ndarray): Input data matrix.
        corr_matrix (numpy.ndarray): Pairwise correlation matrix of the input data.
        threshold (float): Defined threshold for correlation values.
        categorical_features (list or array-like): Indices of categorical features.
        continuous_features (list or array-like): Indices of continuous features.
    
    Returns:
        numpy.ndarray: The cleaned input data matrix after removing correlated features.
        numpy.ndarray: The indices of the kept features (relative to the original data).
        list: Updated indices of categorical features in the filtered data.
        list: Updated indices of continuous features in the filtered data.
    """
    correlated_features = np.where(np.abs(corr_matrix) > threshold)
    features_to_keep = np.arange(data.shape[1])

    # Set to keep track of features we've already considered
    features_removed = set()
    
    for i, j in zip(correlated_features[0], correlated_features[1]):
        if i != j and j not in features_removed:
            if i in features_to_keep and j in features_to_keep:
                # Decide which feature to remove. Here, arbitrarily remove feature 'j'
                features_to_keep = features_to_keep[features_to_keep != j]
                features_removed.add(j)
    
    # Filter the data to keep only the selected features
    filtered_data = data[:, features_to_keep]
    
    # Create a mapping from original indices to new indices
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(features_to_keep)}
    
    # Update categorical_features
    updated_categorical_features = [
        index_mapping[i] for i in categorical_features if i in index_mapping
    ]
    
    # Update continuous_features
    updated_continuous_features = [
        index_mapping[i] for i in continuous_features if i in index_mapping
    ]
    
    return filtered_data, features_to_keep, updated_categorical_features, updated_continuous_features

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

def random_undersample(x_train, y_train, minority_element):
    """
    Perform random undersampling to balance the dataset by reducing the majority class samples.

    Args:
        x_train (numpy.ndarray): Feature matrix of training data.
        y_train (numpy.ndarray): Labels corresponding to x_train.
        minority_element (int or float): The label of the minority class.

    Returns:
        x_resampled (numpy.ndarray): Resampled feature matrix.
        y_resampled (numpy.ndarray): Resampled labels.
    """
    # Identify minority and majority class indices
    minority_indices = np.where(y_train == minority_element)[0]
    majority_indices = np.where(y_train != minority_element)[0]

    # Number of samples in the minority class
    n_minority = len(minority_indices)

    # Randomly select samples from the majority class
    np.random.seed(42)  # For reproducibility
    majority_indices_undersampled = np.random.choice(
        majority_indices, size=n_minority, replace=False
    )

    # Combine minority class with undersampled majority class
    undersampled_indices = np.concatenate([minority_indices, majority_indices_undersampled])

    # Shuffle the indices to mix class samples
    np.random.shuffle(undersampled_indices)

    # Resample the data
    x_resampled = x_train[undersampled_indices]
    y_resampled = y_train[undersampled_indices]

    return x_resampled, y_resampled

def random_oversample(x_train, y_train, minority_element):
    """
    Perform random oversampling to balance the dataset by increasing the minority class samples.

    Args:
        x_train (numpy.ndarray): Feature matrix of training data.
        y_train (numpy.ndarray): Labels corresponding to x_train.
        minority_element (int or float): The label of the minority class.

    Returns:
        x_resampled (numpy.ndarray): Resampled feature matrix.
        y_resampled (numpy.ndarray): Resampled labels.
    """
    # Identify minority and majority class indices
    minority_indices = np.where(y_train == minority_element)[0]
    majority_indices = np.where(y_train != minority_element)[0]

    # Number of samples in each class
    n_minority = len(minority_indices)
    n_majority = len(majority_indices)

    # Calculate the number of additional samples needed
    n_samples_needed = n_majority - n_minority

    # Randomly sample with replacement from the minority class
    np.random.seed(42)  # For reproducibility
    minority_indices_oversampled = np.random.choice(
        minority_indices, size=n_samples_needed, replace=True
    )

    # Combine original minority indices with oversampled indices
    total_minority_indices = np.concatenate([minority_indices, minority_indices_oversampled])

    # Combine majority indices with the new minority indices
    oversampled_indices = np.concatenate([majority_indices, total_minority_indices])

    # Shuffle the indices to mix class samples
    np.random.shuffle(oversampled_indices)

    # Resample the data
    x_resampled = x_train[oversampled_indices]
    y_resampled = y_train[oversampled_indices]

    return x_resampled, y_resampled

def kmeans(X, n_clusters, max_iters=10):
    """
    Memory-efficient KMeans clustering.
    Args:
        X (numpy.ndarray): Data points (N x D).
        n_clusters (int): Number of clusters.
        max_iters (int): Maximum iterations.
    Returns:
        numpy.ndarray: Centroids of clusters.
    """
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centroids = X[indices]

    for iteration in range(max_iters):
        print(f"Iteration {iteration + 1}/{max_iters}")
        cluster_labels = np.zeros(X.shape[0], dtype=np.int32)

        # Assign clusters without full distance matrix
        for i in range(X.shape[0]):
            distances = np.linalg.norm(X[i] - centroids, axis=1)
            cluster_labels[i] = np.argmin(distances)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(n_clusters, dtype=np.int32)

        for i in range(X.shape[0]):
            cluster = cluster_labels[i]
            new_centroids[cluster] += X[i]
            counts[cluster] += 1

        for k in range(n_clusters):
            if counts[k] > 0:
                new_centroids[k] /= counts[k]
            else:
                new_centroids[k] = centroids[k]  # Keep old centroid if no points assigned

        # Check for convergence
        if np.allclose(centroids, new_centroids, atol=1e-4):
            print("Convergence reached.")
            break

        centroids = new_centroids

    return centroids

def combined_over_under_sampling(X, y, majority_class, minority_class, desired_majority_ratio=0.5, random_state=None):
    """
    Balances the dataset by under-sampling the majority class and over-sampling the minority class.
    
    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Label vector of shape (n_samples,). Labels should be binary (e.g., -1 and 1).
        desired_majority_ratio (float): Desired proportion of majority class samples after resampling (between 0 and 1).
        random_state (int, optional): Seed for reproducibility.
    
    Returns:
        X_resampled (numpy.ndarray): Resampled feature matrix.
        y_resampled (numpy.ndarray): Resampled label vector.
    """
    np.random.seed(random_state)


    # Separate majority and minority samples
    X_majority = X[y == majority_class]
    y_majority = y[y == majority_class]
    X_minority = X[y == minority_class]
    y_minority = y[y == minority_class]

    # Compute desired number of samples
    total_samples = len(y)
    n_desired_majority = int(total_samples * desired_majority_ratio)
    n_desired_minority = total_samples - n_desired_majority

    # Under-sample majority class
    if n_desired_majority < len(y_majority):
        indices_majority = np.random.choice(len(y_majority), n_desired_majority, replace=False)
        X_majority_resampled = X_majority[indices_majority]
        y_majority_resampled = y_majority[indices_majority]
    else:
        X_majority_resampled = X_majority
        y_majority_resampled = y_majority

    # Over-sample minority class
    if n_desired_minority > len(y_minority):
        n_samples_to_add = n_desired_minority - len(y_minority)
        indices_minority = np.random.choice(len(y_minority), n_samples_to_add, replace=True)
        X_minority_oversampled = X_minority[indices_minority]
        y_minority_oversampled = y_minority[indices_minority]
        X_minority_resampled = np.vstack((X_minority, X_minority_oversampled))
        y_minority_resampled = np.hstack((y_minority, y_minority_oversampled))
    else:
        indices_minority = np.random.choice(len(y_minority), n_desired_minority, replace=False)
        X_minority_resampled = X_minority[indices_minority]
        y_minority_resampled = y_minority[indices_minority]

    # Combine resampled majority and minority classes
    X_resampled = np.vstack((X_majority_resampled, X_minority_resampled))
    y_resampled = np.hstack((y_majority_resampled, y_minority_resampled))

    # Shuffle the resampled dataset
    indices = np.arange(len(y_resampled))
    np.random.shuffle(indices)
    X_resampled = X_resampled[indices]
    y_resampled = y_resampled[indices]

    return X_resampled, y_resampled

def add_bias_term(x):
    bias = np.ones((x.shape[0], 1))
    return np.hstack((bias, x))