import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def load_and_inspect_data(file_path):
    data = pd.read_csv(file_path)
    
    print("\nValores faltantes por columna:\n", data.isnull().sum())

    print("\nPrimeras filas del dataset:\n", data.head())
    
    return data

def preprocess_data(data):
    imputer = SimpleImputer(strategy='mean')
    data_numeric = data.select_dtypes(include=['float64', 'int64'])
    data[data_numeric.columns] = imputer.fit_transform(data_numeric)

    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    return data, label_encoders

def scale_data(data, feature_columns):
    numeric_columns = data[feature_columns].select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()

    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    return data

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

def train_and_evaluate(X_train, X_test, y_train, y_test, degree=2):
    # Transformar las características en características polinómicas
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    

    regressor = LinearRegression()
    regressor.fit(X_train_poly, y_train)
    
    # Predicciones
    y_pred = regressor.predict(X_test_poly)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nMean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² (Coefficient of Determination): {r2:.2f}")

    plot_residuals(y_test, y_pred)

def main(file_path, degree=2):
    data = load_and_inspect_data(file_path)

    X = data.drop(columns=['Calories_Burned'])
    y = data['Calories_Burned']

    data_processed, encoders = preprocess_data(data)

    feature_columns = X.columns
    data_processed = scale_data(data_processed, feature_columns)

    X = data_processed.drop(columns=['Calories_Burned'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_evaluate(X_train, X_test, y_train, y_test, degree)


main("gym_members_exercise_tracking.csv", degree=2)
