import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv('gym_members_exercise_tracking.csv')

# Step 2: Select features for dimensionality reduction (Max_BPM, Avg_BPM, Resting_BPM)
selected_features = ['Max_BPM', 'Avg_BPM', 'Resting_BPM']
data_selected = data[selected_features]

# Step 3: Standardize the selected features manually
means = data_selected.mean()
stds = data_selected.std()
standardized_data = (data_selected - means) / stds  # Standardize the data

# Step 4: Calculate the covariance matrix of the standardized data
cov_matrix = np.cov(standardized_data, rowvar=False)

# Step 5: Calculate eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 6: Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Step 7: Select the first eigenvector (corresponding to the largest eigenvalue) as the principal component
principal_component = sorted_eigenvectors[:, 0]

# Step 8: Project the standardized data onto the first principal component to create the new feature
pc1 = standardized_data.dot(principal_component)
data['PC1'] = pc1  # Add the new PCA feature to the dataset

# Step 9: Calculate variance for original features and new PCA feature
variances = data[selected_features + ['PC1']].var()

# Step 10: Normalize the variances of the original features (Max_BPM, Avg_BPM, Resting_BPM)
total_variance = variances.sum()
normalized_variances = variances[selected_features] / total_variance

# Step 11: Plot histograms for each original feature and the new PCA feature
fig, axes = plt.subplots(2, 2, figsize=(12, 6))  # Smaller window size
axes = axes.ravel()

# Adjusted the number of bins in the histograms to make them smaller
for i, col in enumerate(selected_features + ['PC1']):
    axes[i].hist(standardized_data[col] if col != 'PC1' else data['PC1'], bins=10, alpha=0.7, color='c')  # Standardized data for comparison
    axes[i].set_title(f'{col} Histogram (Variance: {variances[col]:.2f})')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
    
    # Annotate the histograms with the normalized variance for the original features
    if col in selected_features:
        # Adjusted position of the annotation to move further on top
        axes[i].annotate(f'Normalized Variance: {normalized_variances[col]:.2f}', 
                         xy=(0.5, 1.15), xycoords='axes fraction', 
                         fontsize=10, color='red', ha='center', va='bottom')

# Adjust spacing between plots
plt.subplots_adjust(hspace=1, wspace=1)  # Increased separation between plots

plt.show()
