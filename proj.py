import numpy as np
import os
from TFMLP import MLPR
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale
 
pth = os.path.join(os.getcwd(), "sample2.csv")
A = np.loadtxt(pth, delimiter=",", usecols=(0, 1))
print("A pehla")
print(A)
A = scale(A)
print("A scale ke baad")
print(A)
#y is the dependent variable
y = A[:, 1].reshape(-1, 1)
#A contains the independent variable

A = A[:, 0].reshape(-1, 1)
print("A reshape ke baad")
print(A)
#Plot the high value of the stock price
#mpl.plot(A[:, 0], y[:, 0])
#mpl.show()


#Number of neurons in the input layer
i = 1
#Number of neurons in the output layer
o = 1
#Number of neurons in the hidden layers
h = 32
#The list of layer sizes
layers = [i, h, h, h, h, h, h, h, h, h, o]
mlpr = MLPR(layers, maxItr = 2000, tol = 0.15, reg = 0.1, verbose = True)

#Length of the hold-out period
nDays = 5
n = len(A)
#Learn the data
mlpr.fit(A[0:(n-nDays)], y[0:(n-nDays)])

print("A dega")
print(A)
#Begin prediction
yHat = mlpr.predict(A)
print("Output")
print(A)

#print(B)
#print(mlpr.predict(B))
#Plot the results
mpl.plot(A, y, c='#b0403f')
mpl.plot(A, yHat, c='#000000')


B = np.concatenate([np.arange(A[0], 1.25, 0.01).reshape(-1, 1), A], axis = 0)
mpl.plot(B, mlpr.predict(B), c='#5aa9ab')
mpl.show()