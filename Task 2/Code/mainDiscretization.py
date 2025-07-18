import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = "gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

columns_to_discretize = ['Weight (kg)', 'Height (m)', 'Calories_Burned', 'BMI']

def equal_width_discretization(data, columns, num_bins):
    for col in columns:
        data[f'{col}_EW'] = pd.cut(data[col], bins=num_bins)
    return data

num_bins = 4
data = equal_width_discretization(data, columns_to_discretize, num_bins)

def plot_distributions(data, columns):
    fig, axes = plt.subplots(2, len(columns), figsize=(len(columns) * 5, 10))
    
    for i, col in enumerate(columns):
        axes[0, i].hist(data[col], bins=30, color='blue', alpha=0.7)
        axes[0, i].set_title(f'Original {col}')
        
        bin_counts = data[f'{col}_EW'].value_counts().sort_index()  # Ensure bins are ordered sequentially
        axes[1, i].bar(
            x=bin_counts.index.astype(str),
            height=bin_counts.values,
            color='orange',
            alpha=0.7
        )
        axes[1, i].set_title(f'{col} (Equal Width)')
        axes[1, i].set_xticklabels(bin_counts.index.astype(str), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.show()

plot_distributions(data, columns_to_discretize)

