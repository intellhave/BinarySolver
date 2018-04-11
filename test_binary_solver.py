import numpy as np
from binary_solver import BinarySolver, BruteForceBinarySolver, NextPermute







# Test with a random function min x'Wx

def myFunction(x):
    return -((x[0] - 1) + 5*(x[1]-5)**3 + 9*x[2] + x[3]**2 + 10*x[4]**3 - x[5]) 

x0 = np.array([-1,-1,-1,1,1,1])



x, minCost = BruteForceBinarySolver(myFunction, x0)
print(x)
print (minCost)

x = BinarySolver(myFunction, x0, 100, 1000)
print(x)