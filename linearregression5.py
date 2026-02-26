import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Create dataset


df = pd.read_csv("Delievery_dataset.csv")

print("Dataset:")
print(df)

# Step 2: Features and Target
X = df[["Distance_km", "Items", "Traffic_Level", "Processing_Time_hr"]]
y = df["Delivery_Time_hr"]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.3f}")

print("Intercept:", model.intercept_)

# Step 6: Predictions
y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test.values, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

# Step 7: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# Step 8: Predict delivery time for a new order
new_order = pd.DataFrame({
    "Distance_km": [10],
    "Items": [4],
    "Traffic_Level": [2],
    "Processing_Time_hr": [1.5]
})

predicted_time = model.predict(new_order)

print(f"\nPredicted Delivery Time: {predicted_time[0]:.2f} hours")

# Optional Visualization (Distance vs Delivery Time)
plt.scatter(df["Distance_km"], df["Delivery_Time_hr"], label="Actual Data")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (hours)")
plt.title("Delivery Time vs Distance")
plt.legend()
plt.show()