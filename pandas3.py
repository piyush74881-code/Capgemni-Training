# import pandas as pd

# data = {
#     "Name":["A","B","C"],
#     "Salary":[50000,60000,55000]
# }
# df=pd.DataFrame(data)
# print(df)

import pandas as pd


df=pd.read_csv("Dataset.csv");
print(df)

#Find employees with Score > 90 (Exceptional Performers)
exceptional = df[df["Score"] > 90]
print(exceptional)

#Filter records for the Sales Department only

sales_dept = df[df["Department"] == "Sales"]
print(sales_dept)

#Sort employees by Score (Descending)
ranked = df.sort_values(by="Score", ascending=False)
print(ranked)

#Sort by Department first, then Score
dept_ranking = df.sort_values(by=["Department", "Score"], ascending=[True, False])
print(dept_ranking)

#Average score per Department
avg_department_score = df.groupby("Department")["Score"].mean()
print(avg_department_score)

#Maximum score in each Quarter
max_quarter_score = df.groupby("Quarter")["Score"].max()
print(max_quarter_score)