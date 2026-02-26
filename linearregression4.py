import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Create dataset
df = pd.read_csv("multil_salary_pred.csv")

print("Dataset:")
print(df)

# Step 2: Features and Target
X = df[["Experience_years", "Education_Level", "Skills_Count", "Performance_Rating"]]
y = df["Salary_lpa"]

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

# Step 8: Predict salary for a new employee
new_employee = pd.DataFrame({
    "Experience_years": [5],
    "Education_Level": [2],
    "Skills_Count": [7],
    "Performance_Rating": [4]
})

predicted_salary = model.predict(new_employee)

print(f"\nPredicted salary for the new employee: {predicted_salary[0]:.2f} LPA")

# Optional Visualization (Experience vs Salary)
plt.scatter(df["Experience_years"], df["Salary_lpa"], label="Actual Data")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary (LPA)")
plt.title("Experience vs Salary")
plt.legend()
plt.show()