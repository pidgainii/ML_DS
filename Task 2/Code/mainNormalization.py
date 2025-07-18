import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

file_path = "gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)



columns = ['Weight (kg)', 'Height (m)', 'Max_BPM', 'Calories_Burned', 'BMI']

original_stats = data[columns].describe()

# normalization (z score)
scaler = StandardScaler()
data_normalized = data.copy()
data_normalized[columns] = scaler.fit_transform(data[columns])

normalized_stats = data_normalized[columns].describe()

# Visualization: Original vs Normalized
def plot_feature_distributions(original, normalized, title):
    plt.figure(figsize=(14, 6))
    for i, col in enumerate(columns):
        plt.subplot(2, len(columns), i + 1)
        plt.hist(original[col], bins=30, alpha=0.7, label='Original')
        plt.title(f'{col} (Original)')
        
        plt.subplot(2, len(columns), i + len(columns) + 1)
        plt.hist(normalized[col], bins=30, alpha=0.7, color='orange', label='Normalized')
        plt.title(f'{col} (Normalized)')
        
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_feature_distributions(data, data_normalized, "Z-score Normalization")

# statistics for each of the data (original and nroamlized)
print("Original Data Statistics:\n", original_stats, "\n")
print("\nNormalized Data Statistics:\n", normalized_stats)
