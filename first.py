# import numpy as np
# arr=np.array([10,20,30,40,50])
# print(2*arr)
# arr2=np.array([[1,2,3],[4,5,6]]);
# print(1.10*arr)
# data=np.array([1,2,3,4,5,6])
# print(10+data)

# print(data[1:5])


#day 2
#slicing 
#broadcasting

import numpy as np

sales_day1 = np.array([120, 85, 60])
sales_day2 = np.array([150, 90, 75])

#adding the day1 sale and day2 sale

total_sales = sales_day1 + sales_day2
print(total_sales)

#finding maximum percentage growth 
growth_percentage = ((sales_day2 - sales_day1) / sales_day1) * 100
print(growth_percentage)
#finding maximum percent
maxgrowth=0
for i in growth_percentage:
    maxgrowth=max(i,maxgrowth)
print(maxgrowth)

#converting to fereinheight

temps = np.array([
    [25, 28, 30],
    [22, 24, 26],
    [30, 32, 33],
    [27, 29, 31],
    [20, 21, 23]
])

fahrenheit = temps * 9/5 + 32
print(fahrenheit)

correlation=np.array([1,-1,0])
adjustcorralation=fahrenheit+correlation;
print(adjustcorralation)

#patient bp
bp = np.array([
    [120, 80],
    [135, 85],
    [140, 90],
    [110, 70],
    [125, 75]
])

# 1. Extract systolic values
systolic = bp[:, 0]

# 2. Patients with systolic > 130
high_bp_patients = bp[bp[:, 0] > 130]

# 3. Replace diastolic < 80 with 80
bp[bp[:, 1] < 80, 1] = 80

print("Systolic:", systolic)
print("High BP Patients:\n", high_bp_patients)
print("Updated BP Data:\n", bp)