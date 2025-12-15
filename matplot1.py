import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
arr1=[10,29,83,43,52,60]
label1=["Apple","Banana","Orange","papaya","chill","juice"]
plt.pie(arr1,labels=label1,autopct="%1.1f%%",startangle=90)
plt.title("FRUIT")
plt.show()