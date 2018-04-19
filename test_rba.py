"""
Test RBA performance using the MNIST dataset

@author: dang-khoa
"""

from relaxed_ba_bias import relaxed_ba_bias
import scipy
import numpy as np
from scipy import misc, io
from sklearn.metrics import mean_squared_error

mnist = scipy.io.loadmat('data/mnist_data.mat')

xtrain = mnist['Xtrain']
m, D = xtrain.shape
xtest = mnist['Xtest']

W2, W1, c2, c1, B = relaxed_ba_bias(xtrain, 10, 50.0, 0.1, 300)

xtrain = xtrain.T
H = np.matmul(W1, xtrain) + c1*np.ones((1, m))
Xctr = np.matmul(W2, np.sign(H)) + c2*np.ones((1, m))

print("MSE of training samples =" + str(mean_squared_error(Xctr, xtrain)))
