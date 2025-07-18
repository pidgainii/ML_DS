import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('gym_members_exercise_tracking.csv')
# only numerical columns
numeric_data = data.select_dtypes(include=['number'])

X = numeric_data.drop('Calories_Burned', axis=1)
y = numeric_data['Calories_Burned']

# Normalize numeric data to make it suitable for chi2
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


k = 5  # Select top 5 features
chi2_selector = SelectKBest(chi2, k=k)
X_new_chi2 = chi2_selector.fit_transform(X_scaled, y)

chi2_features = X.columns[chi2_selector.get_support()]

# r-regression
r_selector = SelectKBest(f_regression, k=k)
X_new_r = r_selector.fit_transform(X, y)

r_features = X.columns[r_selector.get_support()]

print("Selected Features Using Chi2:")
print(chi2_features.tolist())

print("\nSelected Features Using R Regression:")
print(r_features.tolist())
