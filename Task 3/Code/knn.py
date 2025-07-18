import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

def load_and_inspect_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Check for missing values
    print("\nMissing values by column:\n", data.isnull().sum())
    
    # Display the first few rows of the dataset
    print("\nFirst rows of the dataset:\n", data.head())
    
    return data

def preprocess_data(data):
    # Impute missing values for numeric columns
    imputer = SimpleImputer(strategy='mean')
    data_numeric = data.select_dtypes(include=['float64', 'int64'])
    data[data_numeric.columns] = imputer.fit_transform(data_numeric)
    
    # Encode categorical variables
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

def plot_confusion_matrix(y_test, y_pred, class_mapping):
    cm = confusion_matrix(y_test, y_pred)
    original_labels = [class_mapping[i] for i in sorted(class_mapping.keys())]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=original_labels, yticklabels=original_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def plot_roc_curve(y_test, y_pred_proba, n_classes, class_mapping):
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        original_class = class_mapping[i]  # Map back to original class
        plt.plot(fpr[i], tpr[i], label=f'Class {original_class} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

def train_and_evaluate(X_train, X_test, y_train, y_test, class_mapping):
    knn = KNeighborsClassifier(n_neighbors=5)  # Example: Using 5 neighbors
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    y_pred_proba = knn.predict_proba(X_test)

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    plot_confusion_matrix(y_test, y_pred, class_mapping)

    n_classes = len(class_mapping)
    plot_roc_curve(y_test, y_pred_proba, n_classes, class_mapping)

def main(file_path):
    data = load_and_inspect_data(file_path)

    X = data.drop(columns=['Experience_Level'])
    y = data['Experience_Level']

    data_processed, encoders = preprocess_data(data)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_mapping = dict(zip(le.transform(le.classes_), le.classes_))
    print("Class Mapping (Encoded to Original):", class_mapping)
    feature_columns = X.columns
    data_processed = scale_data(data_processed, feature_columns)
    X = data_processed.drop(columns=['Experience_Level'])
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    train_and_evaluate(X_train, X_test, y_train, y_test, class_mapping)


main("gym_members_exercise_tracking.csv")
