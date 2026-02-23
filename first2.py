import pandas as pd
import numpy as np

# -------------------------------
# 1. Data Loading
# -------------------------------
customers = pd.read_csv("Customers.csv")
sales = pd.read_csv("Sales.csv")
support = pd.read_csv("Support.csv")

# Inspect datasets
print("Customers:", customers.shape)
print(customers.columns)
print(customers.isnull().sum())

print("\nSales:", sales.shape)
print(sales.columns)
print(sales.isnull().sum())

print("\nSupport:", support.shape)
print(support.columns)
print(support.isnull().sum())

# -------------------------------
# 2. NumPy Array Operations (Broadcasting)
# -------------------------------
prices = sales["Price"].to_numpy()
sales["DiscountedPrice"] = prices * 0.9  # 10% discount

# Revenue per order
sales["Revenue"] = sales["Quantity"] * sales["Price"]


# 3. Indexing & Slicing

sales["OrderDate"] = pd.to_datetime(sales["OrderDate"])

# Orders from January 2025
jan_orders = sales[
    (sales["OrderDate"].dt.month == 1) &
    (sales["OrderDate"].dt.year == 2025)
]

print("\nJanuary 2025 Orders:")
print(jan_orders)

# First 10 rows
first_10_sales = sales.head(10)
print("\nFirst 10 Sales Records:")
print(first_10_sales)


# 4. Filtering

north_customers = customers[customers["Region"] == "North"]
high_revenue_orders = sales[sales["Revenue"] > 10000]

print("\nNorth Region Customers:")
print(north_customers)

print("\nHigh Revenue Orders:")
print(high_revenue_orders)


# 5. Sorting

customers["SignupDate"] = pd.to_datetime(customers["SignupDate"])

sorted_customers = customers.sort_values(by="SignupDate")
sorted_sales = sales.sort_values(by="Revenue", ascending=False)

print("\nSorted Customers:")
print(sorted_customers)

print("\nSorted Sales:")
print(sorted_sales)

# -------------------------------
# 6. Grouping
# -------------------------------
sales_region = sales.merge(customers[["CustomerID", "Region"]], on="CustomerID")

avg_revenue_region = sales_region.groupby("Region")["Revenue"].mean()
print("\nAverage Revenue by Region:")
print(avg_revenue_region)

avg_resolution_issue = support.groupby("IssueType")["ResolutionTime"].mean()
print("\nAverage Resolution Time by Issue Type:")
print(avg_resolution_issue)


# 7. Data Wrangling


# Handle missing Age values
customers["Age"].fillna(customers["Age"].median(), inplace=True)

# Rename columns (if needed)
customers.rename(columns={"CustomerID": "CustomerID"}, inplace=True)
sales.rename(columns={"CustomerID": "CustomerID"}, inplace=True)
support.rename(columns={"CustomerID": "CustomerID"}, inplace=True)

# Merge datasets
merged_data = customers.merge(sales, on="CustomerID").merge(support, on="CustomerID")

print("\nMerged Dataset:")
print(merged_data.head())


# 8. Customer Lifetime Value (CLV)

clv = sales.groupby("CustomerID")["Revenue"].sum().reset_index()
clv.rename(columns={"Revenue": "CLV"}, inplace=True)

# Average Resolution Time per Customer
avg_resolution_customer = support.groupby("CustomerID")["ResolutionTime"].mean().reset_index()
avg_resolution_customer.rename(columns={"ResolutionTime": "AvgResolutionTime"}, inplace=True)

# Merge new fields
final_data = customers.merge(clv, on="CustomerID", how="left")
final_data = final_data.merge(avg_resolution_customer, on="CustomerID", how="left")

print("\nFinal Dataset with CLV and Avg Resolution Time:")
print(final_data.head())


# 9. Export Cleaned Dataset

final_data.to_csv("Cleaned_Data.csv", index=False)

print("\nCleaned dataset exported successfully: Cleaned_Data.csv")