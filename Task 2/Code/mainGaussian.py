import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

file_path = "gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

selected_features = ['Weight (kg)', 'Height (m)', 'Calories_Burned', 'BMI']

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[selected_features])

n_components = 3
gmm = GaussianMixture(n_components=n_components, random_state=42)
data['Cluster'] = gmm.fit_predict(data_scaled)

sns.pairplot(
    data=data,
    vars=selected_features,
    hue='Cluster',
    palette='Set2',
    diag_kind='kde',
    height=2.5
)

plt.suptitle('Gaussian Mixture Model Clustering', y=1.02)
plt.show()
