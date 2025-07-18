import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the data
file_path = "ev_charging_patterns.csv"
df = pd.read_csv(file_path)

categorical_columns = df.select_dtypes(include=['object']).columns
numerical_columns = df.select_dtypes(include=['number']).columns

target_column = 'Charging Duration (hours)'
categorical_columns = categorical_columns.tolist() 
numerical_columns = numerical_columns.tolist()

numerical_columns.remove(target_column)

df_features = df[categorical_columns + numerical_columns]

# encoding categorical features as integer indices using LabelEncoder
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df_features[col] = le.fit_transform(df_features[col])
    label_encoders[col] = le

imputer = SimpleImputer(strategy='mean') 
df_features = imputer.fit_transform(df_features)

y = df[target_column].values 
X = df_features 

# Splitting data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


threshold = 0.5
correct_predictions = np.sum(np.abs(y_pred - y_test) < threshold)
total_predictions = len(y_test)


feature_importances = regressor.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

top_5_features = np.array(categorical_columns + numerical_columns)[sorted_idx][:5]
top_5_importances = feature_importances[sorted_idx][:5]

# Print evaluation metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")
print(f"Number of correct predictions: {correct_predictions}")
print(f"Total number of predictions: {total_predictions}")

print("\n5 most influential features:")
for feature, importance in zip(top_5_features, top_5_importances):
    print(f"{feature}: {importance}")
