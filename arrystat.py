import numpy as np
arry1=np.arange(20)
print("the array is :",arry1)
arry2=arry1.reshape(2,5,2)
print("the reshaped array is :\n",arry2)
flaten1=arry2.ravel()
print("the flatened array is :",flaten1)
 