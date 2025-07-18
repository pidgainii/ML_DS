import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel


df = pd.read_csv('gym_members_exercise_tracking.csv')
df = df.select_dtypes(include=['number'])

y_synthetic = np.random.rand(len(df))  # Generate random values as synthetic target

X = df  # All columns are features

dt_model = DecisionTreeRegressor(random_state=67)
dt_model.fit(X, y_synthetic) # model is being trained (we use y_synthetic because we dont need prediction)

selector = SelectFromModel(dt_model, threshold="mean")
X_selected = selector.transform(X)
selected_features = X.columns[selector.get_support()]

print("Top features selected by Decision Tree model:")
print(selected_features)

importances = dt_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature importance (sorted):")
print(importance_df)
