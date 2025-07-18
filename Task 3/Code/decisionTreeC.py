import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

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

def plot_roc_curve(y_test, y_pred_proba, class_labels):
    y_test_bin = label_binarize(y_test, classes=class_labels)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i, label in enumerate(class_labels):
        fpr[label], tpr[label], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])

    plt.figure(figsize=(10, 8))
    for label in class_labels:
        plt.plot(fpr[label], tpr[label], label=f'Class {label} (AUC = {roc_auc[label]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def train_and_evaluate(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)
    
    correct_predictions = (y_pred == y_test).sum()
    total_predictions = len(y_test)

    print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Correct Predictions: {correct_predictions} out of {total_predictions}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    class_labels = sorted(y_test.unique())
    plot_confusion_matrix(y_test, y_pred, class_labels)
    plot_roc_curve(y_test, y_pred_proba, class_labels)

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
