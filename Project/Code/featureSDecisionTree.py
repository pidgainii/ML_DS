import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

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

# Encoding categorical features as integer indices using LabelEncoder
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

# Print the 5 most influential features
print("\n5 most influential features:")
for feature, importance in zip(top_5_features, top_5_importances):
    print(f"{feature}: {importance}")




# Variance Thresholding
variance_threshold = VarianceThreshold(threshold=0.01)
X_var_thresholded = variance_threshold.fit_transform(X_train)

# Train model with the reduced features
regressor_var_threshold = DecisionTreeRegressor(random_state=42)
regressor_var_threshold.fit(X_var_thresholded, y_train)
y_pred_var_threshold = regressor_var_threshold.predict(variance_threshold.transform(X_test))

# SelectKBest (using f_regression instead of chi2 for regression problem)
select_k_best = SelectKBest(score_func=f_regression, k='all')  # Using f_regression for continuous data
X_k_best = select_k_best.fit_transform(X_train, y_train)

# Train model with the reduced features
regressor_k_best = DecisionTreeRegressor(random_state=42)
regressor_k_best.fit(X_k_best, y_train)
y_pred_k_best = regressor_k_best.predict(select_k_best.transform(X_test))

# R_regression
regressor_r = make_pipeline(LinearRegression())
regressor_r.fit(X_train, y_train)

coef_selector = np.abs(regressor_r.named_steps['linearregression'].coef_)
important_features_r = np.argsort(coef_selector)[::-1]

X_r_selected = X_train[:, important_features_r[:5]]
regressor_r_selection = DecisionTreeRegressor(random_state=42)
regressor_r_selection.fit(X_r_selected, y_train)
y_pred_r_selection = regressor_r_selection.predict(X_test[:, important_features_r[:5]])

selector_model = SelectFromModel(regressor, threshold="mean")
X_selected_model = selector_model.fit_transform(X_train, y_train)

regressor_model_selection = DecisionTreeRegressor(random_state=42)
regressor_model_selection.fit(X_selected_model, y_train)
y_pred_model_selection = regressor_model_selection.predict(selector_model.transform(X_test))


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    threshold = 0.5
    correct_predictions = np.sum(np.abs(y_pred - y_true) < threshold)
    total_predictions = len(y_true)
    
    return mae, mse, r2, correct_predictions, total_predictions

# Variance Thresholding
mae_var_threshold, mse_var_threshold, r2_var_threshold, correct_var_threshold, total_var_threshold = evaluate_model(y_test, y_pred_var_threshold)

# SelectKBest
mae_k_best, mse_k_best, r2_k_best, correct_k_best, total_k_best = evaluate_model(y_test, y_pred_k_best)

# R_regression
mae_r_selection, mse_r_selection, r2_r_selection, correct_r_selection, total_r_selection = evaluate_model(y_test, y_pred_r_selection)

# SelectFromModel
mae_model_selection, mse_model_selection, r2_model_selection, correct_model_selection, total_model_selection = evaluate_model(y_test, y_pred_model_selection)

print("\nVariance Thresholding Results:")
print(f"MAE: {mae_var_threshold}, MSE: {mse_var_threshold}, R2: {r2_var_threshold}")
print(f"Correct predictions: {correct_var_threshold}/{total_var_threshold}")

print("\nSelectKBest Results:")
print(f"MAE: {mae_k_best}, MSE: {mse_k_best}, R2: {r2_k_best}")
print(f"Correct predictions: {correct_k_best}/{total_k_best}")

print("\nR_regression Results:")
print(f"MAE: {mae_r_selection}, MSE: {mse_r_selection}, R2: {r2_r_selection}")
print(f"Correct predictions: {correct_r_selection}/{total_r_selection}")

print("\nSelectFromModel Results:")
print(f"MAE: {mae_model_selection}, MSE: {mse_model_selection}, R2: {r2_model_selection}")
print(f"Correct predictions: {correct_model_selection}/{total_model_selection}")
