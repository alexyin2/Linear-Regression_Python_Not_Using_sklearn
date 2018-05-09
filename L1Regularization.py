# L1 Regression
# L1 Regression is also knowned as LASSO Regression

"""
In my opinion, I think L1 Regression are often used when we have too many variables, 
and we may only want to pick some of the variables.

The goal of L1 Regularization is to select a small number of important features that predict the trend.

We add absolution of weights(parameters) times lambda, which is a constant into the model.

In the generated weights(parameters), some variables may have small values and 
we may pick those variables out.

Warning: The variables with some values doesn't mean that it's not important, but we can say it has less contribution to our model.

It is also pretty important that we can't solve LASSO Regression by OLS, and we'll have to use Gradient Descent to find our solution.
"""
import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50

X = (np.random.random((N, D)) - 0.5) * 10
print(X)

true_w = np.array([1, 0.5, -0.5] + [0] * (D-3))

Y = X.dot(true_w) + np.random.randn(N) * 0.5
print(Y)

# Run gradient descent instead
# 1. Create a list to store the cost in each step
costs = []
# 2. initialize random weights(parameters)
w = np.random.randn(D) / np.sqrt(D)
# 3. Set learning rate
learning_rate = 0.001
# 4. Set lambda
l1 = 10.0
# 5. Run Gradient Descend
for _ in range(500):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate * (X.T.dot(delta) + l1*np.sign(w))
    mse = delta.dot(delta) / N
    costs.append(mse)
# 6. plot the costs
plt.plot(costs)
plt.show()
#7. plot true w and the w we found
plt.plot(true_w, 'b,', label='true w')
plt.plot(w, 'r.', label='w map')
plt.legend()
plt.show()

# What if we use linear regression to calculate the weights
w_linear = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
plt.plot(true_w, 'b,', label='true w')
plt.plot(w_linear, 'r.', label='w linear')
plt.legend()
plt.show()
