import pandas as pd
from sklearn.impute import SimpleImputer

# Load the data
file_path = "ev_charging_patterns.csv"
df = pd.read_csv(file_path)

missing_values = df[df.isnull().any(axis=1)]
print("Rows with missing values:")
print(missing_values)

df_no_missing_rows = df.dropna(axis=0) 
print("\nData after removing rows with missing values:")
print(df_no_missing_rows.head())


# Impute missing values for numerical columns with the mean
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
imputer_mean = SimpleImputer(strategy='mean')
df_imputed_mean = df.copy()
df_imputed_mean[numerical_columns] = imputer_mean.fit_transform(df[numerical_columns])
print(df_imputed_mean.head())

# Impute missing values for categorical columns with the mode
categorical_columns = df.select_dtypes(include=['object']).columns
imputer_mode = SimpleImputer(strategy='most_frequent')
df_imputed_mode = df.copy()
df_imputed_mode[categorical_columns] = imputer_mode.fit_transform(df[categorical_columns])
print(df_imputed_mode.head())
