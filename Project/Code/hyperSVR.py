import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from scipy.stats import uniform

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

imputer = SimpleImputer(strategy='mean')  # Replace NaN with the mean of each column
df_features = imputer.fit_transform(df_features)

y = df[target_column].values 
X = df_features 

# Splitting the data into 80% training, 10% hyperparameter search, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.05, random_state=42)
X_search, X_test, y_search, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

svr = SVR()

param_distributions = {
    'kernel': ['linear', 'rbf'], 
    'C': uniform(0.1, 100), 
    'epsilon': uniform(0.01, 0.5),
    'gamma': ['scale', 'auto']
}

random_search = RandomizedSearchCV(
    estimator=svr, 
    param_distributions=param_distributions, 
    n_iter=20,  # Try 20 random combinations
    cv=2,  # Reduce cross-validation folds to 2
    n_jobs=-1, 
    verbose=1, 
    scoring='neg_mean_squared_error',
    random_state=42
)

random_search.fit(X_search, y_search)

best_svr = random_search.best_estimator_

y_pred = best_svr.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

threshold = 0.5
correct_predictions = np.sum(np.abs(y_pred - y_test) < threshold)
total_predictions = len(y_test)

print(f"Best Hyperparameters: {random_search.best_params_}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")
print(f"Number of correct predictions: {correct_predictions}")
print(f"Total number of predictions: {total_predictions}")
