import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

columns = ['Weight (kg)', 'Height (m)', 'Max_BPM', 'Calories_Burned', 'BMI']

# Handle negative values
data_log_transformed = data.copy()
for col in columns:
    # If there are non-positive values, add a small constant (e.g., 1)
    if (data[col] <= 0).any():
        data_log_transformed[col] = np.log(data[col] + 1)
    else:
        data_log_transformed[col] = np.log(data[col])


def plot_feature_distributions(original, transformed, title):
    plt.figure(figsize=(14, 8))
    for i, col in enumerate(columns):
        # Original Distribution
        plt.subplot(2, len(columns), i + 1)
        plt.hist(original[col], bins=30, alpha=0.7, label='Original')
        plt.title(f'{col} (Original)')
        
        # Log-Transformed Distribution
        plt.subplot(2, len(columns), i + len(columns) + 1)
        plt.hist(transformed[col], bins=30, alpha=0.7, color='orange', label='Log-Transformed')
        plt.title(f'{col} (Log-Transformed)')
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_feature_distributions(data, data_log_transformed, "Log Transformation of Features")
