import numpy as np
from matplotlib import pyplot as plt

X = ['Group A','Group B','Group C','Group D']
Ygirls = np.array([10,20,20,40])
Yboys = np.array([20,30,25,30])
Yothers = np.array([15,35,20,35])

Y2girls = np.array([15,35,20,35])
Y2boys = np.array([10,20,20,40])
Y2others = np.array([20,30,25,30])

Zgirls = 90 - Ygirls - Y2girls
Zboys = 90 - Yboys - Y2boys
Zothers = 90 - Yothers - Y2others
  
X_axis = np.arange(len(X))

bottom_girls = np.zeros_like(Ygirls)
bottom_boys = np.zeros_like(Yboys)
bottom_others = np.zeros_like(Yothers)
  
plt.bar(X_axis - 0.20, Ygirls, width=0.15, label='bottom', bottom=bottom_girls, color='#2CD23E')
plt.bar(X_axis - 0.00, Yothers, width=0.15, bottom=bottom_others, color='#2CD23E')
plt.bar(X_axis + 0.20, Yboys, width=0.15, bottom=bottom_boys, color='#2CD23E')

bottom_girls += Ygirls
bottom_boys += Yboys
bottom_others += Yothers

plt.bar(X_axis - 0.20, Y2girls, width=0.15, label='middle', bottom=bottom_girls, color='#E53629')
plt.bar(X_axis - 0.00, Y2others, width=0.15, bottom=bottom_others, color='#E53629')
plt.bar(X_axis + 0.20, Y2boys, width=0.15, bottom=bottom_boys, color='#E53629')

bottom_girls += Y2girls
bottom_boys += Y2boys
bottom_others += Y2others

plt.bar(X_axis - 0.20, Zgirls, width=0.15, label='above', bottom=bottom_girls, color='#4149C3')
plt.bar(X_axis - 0.00, Zothers, width=0.15, bottom=bottom_others, color='#4149C3')
plt.bar(X_axis + 0.20, Zboys, width=0.15, bottom=bottom_boys, color='#4149C3')
  
plt.xticks(X_axis, X)
plt.xlabel("Groups")
plt.ylabel("Number of Students")
plt.title("Number of Students in each group")
plt.legend()
plt.show()