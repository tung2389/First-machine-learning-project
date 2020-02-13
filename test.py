import numpy as np
import os

X = [np.zeros((6,4)),np.zeros(6)]

X[0] = [[1,2,3,4],[5,6,7,8]]
X[1] = [1,2,3,4]
cwd = os.getcwd()
with open(os.getcwd() + '/Logistic regression/test.csv','r') as file:
    print(np.loadtxt(file))