import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Loads CSV file
    dataframe = pd.read_csv('spotify-2023.csv', encoding = 'latin')

    # Calculates statistics only for numerical columns
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns  # Seleccionar solo columnas numéricas

    # Calculates and prints statistics
    mean_values = dataframe[numeric_columns].mean()
    median_values = dataframe[numeric_columns].median()
    variance_values = dataframe[numeric_columns].var()

    print("Mean:\n", mean_values.to_string(), "\n")
    print("Median:\n", median_values.to_string(), "\n")
    print("Variance:\n", variance_values.to_string(), "\n")


    # Histogram of numerical columns
    dataframe.hist(bins=20, figsize=(10, 8))
    plt.suptitle('Distribution of numerical characteristics')
    
    # Adjusts vertical space between schemes
    plt.subplots_adjust(hspace=0.7)
    plt.show()
    

    # Calculate covariance matrix for all numerical columns
    cov_matrix = dataframe[numeric_columns].cov()
    print("Covariance Matrix:\n", cov_matrix, "\n")

    # If you want to calculate covariance between two specific columns
    if 'energy_%' in dataframe.columns and 'danceability_%' in dataframe.columns:
        cov_energy_danceability = dataframe['energy_%'].cov(dataframe['danceability_%'])
        print(f"Covariance between energy and danceability: {cov_energy_danceability}\n")

    # Scatter plot between spotify playlists and charts (variables can be changed as wanted)
    plt.scatter(dataframe['in_spotify_playlists'], dataframe['in_spotify_charts'])
    plt.title('Scatter Plot of Spotify Playlists vs Charts')
    plt.xlabel('Playlists')
    plt.ylabel('Charts')
    plt.show()

    correlation_matrix = dataframe[numeric_columns].corr()

    # Correlation heatmap
    plt.figure(figsize=(12, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)   
    plt.xticks(rotation=45, ha='right')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()



except Exception as e:
    print(f"An error ocurred: {e}")







