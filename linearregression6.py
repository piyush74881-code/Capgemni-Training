import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Create dataset
data = {
    "Engine_Size": [1.2,1.5,1.8,2.0,2.2,1.3,1.6,2.4,2.0,1.4,1.7,2.5,1.8,2.2,1.5],
    "Mileage": [90,70,60,50,40,85,65,30,45,80,55,25,50,35,75],
    "Age": [8,6,5,4,3,7,6,2,4,7,5,1,3,2,6],
    "Horsepower": [80,95,110,130,150,85,100,180,140,90,115,200,125,160,105],
    "Price": [3.5,5,6,8,10,4,5.5,14,9,4.5,6.5,16,8.5,12,5.2]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# Step 2: Basic EDA

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nCorrelation Matrix:")
print(df.corr())


# Step 3: Visualization (EDA)

plt.scatter(df["Engine_Size"], df["Price"])
plt.xlabel("Engine Size (Liters)")
plt.ylabel("Price")
plt.title("Engine Size vs Price")
plt.show()

plt.scatter(df["Mileage"], df["Price"])
plt.xlabel("Mileage (thousand km)")
plt.ylabel("Price")
plt.title("Mileage vs Price")
plt.show()

plt.scatter(df["Age"], df["Price"])
plt.xlabel("Car Age (years)")
plt.ylabel("Price")
plt.title("Age vs Price")
plt.show()

plt.scatter(df["Horsepower"], df["Price"])
plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Horsepower vs Price")
plt.show()


# Step 4: Features and Target

X = df[["Engine_Size", "Mileage", "Age", "Horsepower"]]
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Model

model = LinearRegression()
model.fit(X_train, y_train)

# Model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.3f}")

print("Intercept:", model.intercept_)


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

# Step 8: Predict price of a new car

new_car = pd.DataFrame({
    "Engine_Size": [2.0],
    "Mileage": [40],
    "Age": [3],
    "Horsepower": [140]
})

predicted_price = model.predict(new_car)

print(f"\nPredicted Car Price: {predicted_price[0]:.2f} lakhs")