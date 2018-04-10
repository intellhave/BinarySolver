# Testing python optimizer

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.linalg import norm

k = 100
def func(x):
   # return np.array([10*(x[1] - x[0]**2), (1-x[0])])
   return (10*(x[1] - x[0]**2))**2 + (k-x[0])**2

def norm_2(x):
    return norm(x)


x0 = np.array([2,2])
v0 = np.zeros(x0.shape)

k = 100
constrs = ({'type':'ineq',
            'fun': lambda x: np.array([k - norm(x)]),
            'jac': lambda x: np.array(-2*x)
           })

bnds = [-1,1]
for i in range(1):
    bnds = (bnds, [-1,1])



#res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
res_1 = minimize(func, x0, bounds=bnds, constraints=constrs)
#res_1 = least_squares(fun_rosenbrock,x0_rosenbrock, bounds=([-np.inf, 1.5],np.inf))

print(res_1)
