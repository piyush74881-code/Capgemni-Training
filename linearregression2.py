import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load dataset from link

df = pd.read_csv("salary_lpa.csv")

print("Dataset Preview:")
print(df.head())

# Step 2: Define features and target
X = df[["Experience_years"]]
y = df["Salary_lpa"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model parameters
print("\nSlope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

# Step 6: Predictions
y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test.values, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

# Step 7: Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error:", mae)
print("R2 Score:", r2)

# Step 8: Predict salary for new experience
new_experience = pd.DataFrame({"Experience_years": [5]})
predicted_salary = model.predict(new_experience)

print(f"\nPredicted salary for 5 years experience: {predicted_salary[0]:.2f}")

# Step 9: Visualization
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction Model")
plt.legend()
plt.show()