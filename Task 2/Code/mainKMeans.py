import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

file_path = "gym_members_exercise_tracking.csv"  # Replace with the actual file path
data = pd.read_csv(file_path)

selected_features = ['Weight (kg)', 'Height (m)', 'Calories_Burned', 'BMI']  # Customize these features
selected_data = data[selected_features]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)

num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

def plot_clusters(data, cluster_col, selected_features):
    sns.pairplot(data[selected_features + [cluster_col]], hue=cluster_col, palette='tab10')
    plt.suptitle("KMeans Clustering Visualization (Selected Features)", y=1.02)
    plt.show()

plot_clusters(data, 'Cluster', selected_features)

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=selected_features)
print("Cluster Centers:\n", cluster_centers_df)

print("\nCluster Counts:\n", data['Cluster'].value_counts())
