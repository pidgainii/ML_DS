import pandas as pd

# Load the CSV file
df = pd.read_csv('spotify-2023.csv', encoding='latin')

# Check for missing values in each column
missing_per_column = df.isnull().sum()
print("Missing values per column:")
print(missing_per_column[missing_per_column > 0])

# Display rows with any missing values
rows_with_missing_values = df[df.isnull().any(axis=1)]
print("\nRows with missing values:")
print(rows_with_missing_values)

total_missing = df.isnull().sum().sum()
print(f"\nTotal number of missing values in the dataset: {total_missing}")
