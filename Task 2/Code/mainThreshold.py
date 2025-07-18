import pandas as pd
from sklearn.feature_selection import VarianceThreshold

data = pd.read_csv('gym_members_exercise_tracking.csv')

# only numerical columns
numeric_data = data.select_dtypes(include=['number'])

# threshold can be changed
threshold = 1
selector = VarianceThreshold(threshold=threshold)
selector.fit(numeric_data)
selected_features_mask = selector.get_support()

selected_features = numeric_data.columns[selected_features_mask]

print("Selected Features Based on Variance Threshold:")
print(selected_features.tolist())

# reduced data set
reduced_data = numeric_data[selected_features]
print("\nReduced Dataset (First 5 Rows):")
print(reduced_data.head())
