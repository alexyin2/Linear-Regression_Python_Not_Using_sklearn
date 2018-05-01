import numpy as np
import matplotlib.pyplot as plt
"""
L2 Regularization is a way of providing data being highly affected by outliers.

We add squared magnitude of weigths times a constant to our cost function.

This is because large weights may be a sign of overfitting.

L2 Regularization is also called "Ridge Regression".
"""

N = 50
X = np.linspace(0, 10, N)
Y = 0.5 * X + np.random.randn(N)
# Create some outlier
Y[-1] += 100
Y[-2] += 100
# Plot our data
plt.scatter(X, Y)
plt.show()

X = np.vstack([np.ones(N), X]).T

# 1. Linear Regression
# Calculate weights by maximum likelihood 
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
# Plot the scatter plot and Yhat, we can notice that our linear regression is effected by outliers.
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml)
plt.show()

# 2. L2 Regularization
l2 = 1000.0
w_map = np.linalg.solve(l2 * np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)
# Plot the scatter plot with the comparison of two different methods.
# We can notice that L2 Regularization are less effected by outliers.
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Yhat_ml, label='maximum likelihood')
plt.plot(X[:, 1], Yhat_map, label='map')
plt.legend()
plt.show()


















































