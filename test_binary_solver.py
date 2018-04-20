import numpy as np
from binary_solver import BinarySolver, BruteForceBinarySolver, NextPermute







# Test with a random function min x'Wx

x0 = -1*np.ones(1000)
dim = len(x0)
W = np.random.rand(dim, dim)
b = np.random.rand(dim, 1)

def myFunction(x):
    xx  = x[None, :]
    return np.squeeze(xx.dot(W).dot(xx.T)) + xx.dot(b)

# sol, minCost = BruteForceBinarySolver(myFunction, x0)
# print('------BRUTEFORCE SOLUTION--------')
# print(sol)
# print (minCost[0][0])



print('------EXACT PENALTY--------')
rho = 0.1
maxIter =  1000
x = BinarySolver(myFunction, x0, rho, maxIter)
print(x)