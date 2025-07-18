import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

file_path = "gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

features = ['BMI', 'Calories_Burned']
X = data[features]

iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)  # contamination=percentage of expected outliers
outliers = iso_forest.fit_predict(X)

data['Isolation_Outlier'] = outliers  # -1 for outliers, 1 for inliers
data['Isolation_Score'] = iso_forest.decision_function(X)  # Lower score = higher chance of being an outlier

plt.figure(figsize=(8, 6))
plt.scatter(data['BMI'][data['Isolation_Outlier'] == 1], 
            data['Calories_Burned'][data['Isolation_Outlier'] == 1], 
            color='yellow', label='Inliers')
plt.scatter(data['BMI'][data['Isolation_Outlier'] == -1], 
            data['Calories_Burned'][data['Isolation_Outlier'] == -1], 
            color='black', label='Outliers')
plt.xlabel('BMI')
plt.ylabel('Calories Burned')
plt.title('Outlier Detection with Isolation Forest (Yellow and Black)')
plt.legend()
plt.show()
