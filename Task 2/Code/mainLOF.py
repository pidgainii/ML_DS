import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

file_path = "gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

features = ['BMI', 'Calories_Burned']
X = data[features]

lof = LocalOutlierFactor(n_neighbors=20)

outliers = lof.fit_predict(X)

data['LOF_Outlier'] = outliers
data['LOF_Score'] = -lof.negative_outlier_factor_

plt.figure(figsize=(8, 6))

# innliers
plt.scatter(data['BMI'][data['LOF_Outlier'] == 1], 
            data['Calories_Burned'][data['LOF_Outlier'] == 1], 
            color='blue', label='Inliers')

#outliers
plt.scatter(data['BMI'][data['LOF_Outlier'] == -1], 
            data['Calories_Burned'][data['LOF_Outlier'] == -1], 
            color='red', label='Outliers')

plt.xlabel('BMI')
plt.ylabel('Calories Burned')
plt.title('Outlier Detection using Local Outlier Factor (LOF)')
plt.legend()
plt.show()
