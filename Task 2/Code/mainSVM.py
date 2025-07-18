import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

file_path = "gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

features = ['BMI', 'Calories_Burned']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='auto')  # 'nu' controls the proportion of outliers

ocsvm.fit(data_scaled)

outliers = ocsvm.predict(data_scaled)

data['Outlier'] = outliers
plt.figure(figsize=(10, 6))
plt.scatter(data[features[0]], data[features[1]], c=data['Outlier'], cmap='coolwarm', label='Inliers')

plt.scatter(data.loc[data['Outlier'] == -1, features[0]], 
            data.loc[data['Outlier'] == -1, features[1]], 
            color='red', label='Outliers', s=100, edgecolors='black')

plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title("Outlier Detection using One-Class SVM")
plt.legend()
plt.show()