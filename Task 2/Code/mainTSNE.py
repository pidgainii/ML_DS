import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


data = pd.read_csv('gym_members_exercise_tracking.csv')



selected_features = ['Weight (kg)', 'Height (m)', 'BMI']
data_selected = data[selected_features]




data_standardized = (data_selected - data_selected.mean()) / data_selected.std()

tsne = TSNE(n_components=1, perplexity=30, random_state=42)
new_feature = tsne.fit_transform(data_standardized)
data['t-SNE_Feature'] = new_feature

variances = data[selected_features].var()  # Variances of original features
new_feature_variance = data['t-SNE_Feature'].var()  # Variance of the new feature

# Normalized
total_variance = variances.sum() + new_feature_variance
normalized_variances = variances / total_variance
normalized_new_feature_variance = new_feature_variance / total_variance



fig, axes = plt.subplots(1, len(selected_features) + 1, figsize=(15, 5))  # Create subplots
axes = axes.ravel()
for i, col in enumerate(selected_features):
    axes[i].hist(data[col], bins=20, alpha=0.7, color='blue')
    axes[i].set_title(f'{col} (Var: {variances[col]:.2f})')
    axes[i].annotate(f'Norm. Var: {normalized_variances[col]:.2f}',
                     xy=(0.5, 0.9), xycoords='axes fraction', fontsize=10, color='red', ha='center')

# Plot histogram for new feature
axes[-1].hist(data['t-SNE_Feature'], bins=20, alpha=0.7, color='green')
axes[-1].set_title(f't-SNE Feature (Var: {new_feature_variance:.2f})')
axes[-1].annotate(f'Norm. Var: {normalized_new_feature_variance:.2f}',
                  xy=(0.5, 0.9), xycoords='axes fraction', fontsize=10, color='red', ha='center')


plt.tight_layout()
plt.show()
