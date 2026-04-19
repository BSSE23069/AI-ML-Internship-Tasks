import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

# Libraries for Machine Learning
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. LOAD DATASET DIRECTLY FROM LIBRARY (Kaggle-standard data)
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target # Target is in units of $100,000

# 2. EXPLORATORY DATA ANALYSIS (EDA)
print("--- Dataset Overview ---")
print(df.head())

# Correlation Heatmap to see how features like 'AveRooms' affect 'Price'
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', fmt=".2f")
plt.title("Feature Correlation Heatmap (California Housing)")
plt.show()

# 3. PREPROCESSING
# Features: AveRooms (Size), AveBedrms (Bedrooms), Lat/Long (Location)
X = df.drop('Price', axis=1)
y = df['Price']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data (Standardizing features for better model performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. MODEL TRAINING (Gradient Boosting)
print("\nTraining the Gradient Boosting model... please wait.")
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. EVALUATION
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Mean Absolute Error: {mae:.4f} (units of $100k)")
print(f"RMSE: {rmse:.4f}")
print(f"R-Squared Score: {r2:.4f}")

# 6. VISUALIZATION: ACTUAL VS PREDICTED
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Actual Prices ($100k)")
plt.ylabel("Predicted Prices ($100k)")
plt.title("Actual vs. Predicted House Prices")
plt.show()
joblib.dump(model, 'house_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model Saved Successfully!")