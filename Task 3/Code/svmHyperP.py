import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

def load_and_inspect_data(file_path):
    data = pd.read_csv(file_path)
    print("\nMissing values per column:\n", data.isnull().sum())
    print("\nFirst few rows of the dataset:\n", data.head())
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
    scaler = StandardScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data

def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_test, y_pred_proba, n_classes):
    y_test_bin = label_binarize(y_test, classes=range(1, n_classes + 1))  # Start from class 1
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i+1} (AUC = {roc_auc[i]:.2f})')  # Start class from 1

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Define the SVC model
    svm = SVC(probability=True, random_state=42)
    
    # Define hyperparameters for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10], 
        'kernel': ['linear', 'rbf'], 
        'gamma': ['scale', 'auto'], 
        'degree': [3, 4], 
        'class_weight': [None, 'balanced']
    }
    
    # Perform GridSearchCV to find the best parameters
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Hyperparameters found by GridSearchCV:\n{grid_search.best_params_}")
    
    # Train model with the best parameters
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)
    y_pred_proba = best_svm.predict_proba(X_test)
    
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    correct_predictions = (y_pred == y_test).sum()
    print(f"Correct Predictions: {correct_predictions} out of {len(y_test)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    labels = sorted(y_test.unique())
    plot_confusion_matrix(y_test, y_pred, labels)
    n_classes = len(labels)
    plot_roc_curve(y_test, y_pred_proba, n_classes)

def main(file_path):
    data = load_and_inspect_data(file_path)
    X = data.drop(columns=['Experience_Level'])
    y = data['Experience_Level']
    data_processed, encoders = preprocess_data(data)
    feature_columns = X.columns
    data_processed = scale_data(data_processed, feature_columns)
    X = data_processed.drop(columns=['Experience_Level'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_and_evaluate(X_train, X_test, y_train, y_test)

main("gym_members_exercise_tracking.csv")
