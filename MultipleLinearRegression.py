# Deep Learning Prerequisites: Linear Regression in Python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# 1. Multiple Linear Regression
# Load the data
X = []
Y = []
for line in open('/Users/alex/Desktop/DeepLearningPrerequisites/Git_Code&Dataset/machine_learning_examples/linear_regression_class/data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1])  # Adding 1 for b0
    Y.append(float(y))

# Turn X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)
print(X)

# Plot the data in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

# Calculate weights(parameters)
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# Compute r-square
d1 = Y - Yhat
d2 = Y - Y.mean()
SSE = d1.dot(d1)
SST = d2.dot(d2)
r_square = 1 - SSE / SST
print(r_square)


# 2. Polynomial Regression
# Load the data
X = []
Y = []
for line in open('/Users/alex/Desktop/DeepLearningPrerequisites/Git_Code&Dataset/machine_learning_examples/linear_regression_class/data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

# Turn X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)
print(X)
# Plot the data 
plt.scatter(X[:, 1], Y)
plt.show()
# Calculate weights(parameters)
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)
# Plot it all 
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat))
plt.show()
# Compute r-square
d1 = Y - Yhat
d2 = Y - Y.mean()
SSE = d1.dot(d1)
SST = d2.dot(d2)
r_square = 1 - SSE / SST
print(r_square)


# Exercise
df = pd.read_excel('/Users/alex/Desktop/DeepLearningPrerequisites/Git_Code&Dataset/machine_learning_examples/linear_regression_class/mlr02.xls')
X = df.as_matrix()
plt.scatter(X[:, 1], X[:, 0])
plt.show()

plt.scatter(X[:, 2], X[:, 0])
plt.show()
df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]
X2only = df[['X2', 'ones']]
X3only = df[['X3', 'ones']]

def get_r2(X, Y):
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    Yhat = np.dot(X, w)
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    SSE = d1.dot(d1)
    SST = d2.dot(d2)
    r_square = 1 - SSE / SST
    return r_square
print('r_square for X2 only: ', get_r2(X2only, Y))
print('r_square for X3 only: ', get_r2(X3only, Y))
print('r_square for adding both variables: ', get_r2(X, Y))
