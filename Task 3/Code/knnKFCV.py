import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
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
        plt.plot(fpr[i], tpr[i], label=f'Class {i+1} (AUC = {roc_auc[i]:.2f})')  # Show class starting from 1

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def train_and_evaluate_kfold_single_metrics(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all = []
    y_pred_all = []
    y_pred_proba_all = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        y_pred_proba = knn.predict_proba(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_pred_proba_all.extend(y_pred_proba)

    y_true_all = pd.Series(y_true_all)
    y_pred_all = pd.Series(y_pred_all)
    y_pred_proba_all = pd.DataFrame(y_pred_proba_all)

    # Compute metrics
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, average='weighted')
    recall = recall_score(y_true_all, y_pred_all, average='weighted')
    f1 = f1_score(y_true_all, y_pred_all, average='weighted')
    correct_predictions = (y_pred_all == y_true_all).sum()
    total_predictions = len(y_true_all)

    print(f"\nCorrect Predictions: {correct_predictions} out of {total_predictions}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # Confusion Matrix
    print("\nConfusion Matrix:\n", confusion_matrix(y_true_all, y_pred_all))
    print("\nClassification Report:\n", classification_report(y_true_all, y_pred_all))

    # Plot Confusion Matrix and ROC Curve
    labels = sorted(y_true_all.unique())
    plot_confusion_matrix(y_true_all, y_pred_all, labels)

    n_classes = len(labels)
    plot_roc_curve(y_true_all, y_pred_proba_all.to_numpy(), n_classes)

def main(file_path):
    data = load_and_inspect_data(file_path)
    X = data.drop(columns=['Experience_Level'])
    y = data['Experience_Level']
    data_processed, encoders = preprocess_data(data)
    feature_columns = X.columns
    data_processed = scale_data(data_processed, feature_columns)
    X = data_processed.drop(columns=['Experience_Level'])
    train_and_evaluate_kfold_single_metrics(X, y, n_splits=5)

main("gym_members_exercise_tracking.csv")
