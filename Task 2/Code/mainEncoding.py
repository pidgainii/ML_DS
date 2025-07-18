import pandas as pd

file_path = "gym_members_exercise_tracking.csv"
data = pd.read_csv(file_path)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
one_hot_encoded_data = pd.get_dummies(data, columns=categorical_columns)

def generate_encoding_chart(data, one_hot_encoded_data, categorical_columns):
    for col in categorical_columns:
        print(f"=== One-Hot Encoding for '{col}' ===")
        unique_values = data[col].unique()
        
        encoded_columns = [c for c in one_hot_encoded_data.columns if c.startswith(col)]

        chart = pd.DataFrame({'Original Category': unique_values, 'Encoded Columns': encoded_columns})
        print(chart, "\n")
        
generate_encoding_chart(data, one_hot_encoded_data, categorical_columns)
