import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

file_path = "gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

selected_features = ['Weight (kg)', 'Height (m)', 'Calories_Burned', 'BMI']

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[selected_features])

dbscan = DBSCAN(eps=0.5, min_samples=5)
data['Cluster'] = dbscan.fit_predict(data_scaled)

data['Cluster'] = data['Cluster'].apply(lambda x: 'Noise' if x == -1 else x)

sns.pairplot(
    data=data,
    vars=selected_features,
    hue='Cluster',
    palette='Set1',
    diag_kind='kde',
    height=2.5
)

plt.suptitle('DBSCAN Clustering', y=1.02)
plt.show()

print(data[['Weight (kg)', 'Height (m)', 'Calories_Burned', 'BMI', 'Cluster']].head())
