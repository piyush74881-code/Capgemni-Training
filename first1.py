import numpy as np

# yields = np.array([
#     [2800, 3200, 3000, 2900],  # Wheat
#     [3500, 3700, 3600, 3400],  # Rice
#     [2500, 2700, 2600, 2800]   # Corn
# ])

# #increase all yeild by 10 percent
# increasedyield=yields* 1.10
# print(increasedyield)

# #extracrting wheat yield
# wheatyield=increasedyield[0:1,:]
# print("wheatyieldis\n",wheatyield)

# #higher yielsd than 3000
# highyield=increasedyield[increasedyield>3000]
# print("higher yield",highyield)

import numpy as np

production = np.array([
    [450, 520, 480, 500],  # Gadgets
    [600, 650, 700, 620],  # Tools
    [400, 420, 450, 470]   # Machines
])

improved_production = production * 1.05
print(improved_production)

gadgets_production = improved_production[0]
print(gadgets_production)

avg_production = improved_production.mean(axis=1)
print(avg_production)

high_production = improved_production[improved_production > 500]
print(high_production)