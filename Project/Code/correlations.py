import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

file_path = "ev_charging_patterns.csv"
df = pd.read_csv(file_path)

df['Charging Duration (hours)'] = pd.to_numeric(df['Charging Duration (hours)'], errors='coerce')

numerical_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_df.corr(method='pearson')
charging_duration_correlation = correlation_matrix['Charging Duration (hours)'].sort_values(ascending=False)
top_3_correlated = charging_duration_correlation[1:4]
print("Top 3 Correlated Variables with 'Charging Duration (hours)' (Pearson's Correlation):")
print(top_3_correlated)

# Categorical variables Chi-squared test
categorical_df = df.select_dtypes(include=['object'])

categorical_df = categorical_df.drop(columns=['Charging Start Time', 'Charging End Time', 'User ID'], errors='ignore')

chi_squared_results = {}

for col in categorical_df.columns:
    contingency_table = pd.crosstab(df['Charging Duration (hours)'], df[col])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    chi_squared_results[col] = p

chi_squared_results = {k: v for k, v in sorted(chi_squared_results.items(), key=lambda item: item[1])}
print("\nTop 5 Chi-Squared Test Results (p-values) for Categorical Variables:")
top_5_chi_squared = list(chi_squared_results.items())[:5]
for var, p_value in top_5_chi_squared:
    print(f"{var}: {p_value}")

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Pearson's Correlation Matrix with 'Charging Duration (hours)'")
plt.show()
