# Deep Learning Prerequisites: Linear Regression in Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data with using pandas
X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# Turn X and Y into numpy arrays
X = np.array(X)
Y = np.array(Y)
print(X)

# Plot to see what it looks like
plt.scatter(X, Y)
plt.show()

# Apply the equations we learned to calculate a and b
# This is the mathematical proof of calculating paremeters in simple linear regression based on least square method
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# Calculated predicted Y
Yhat = a * X + b

# Plot it all
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# Calculating r-square
d1 = Y - Yhat
d2 = Y - Y.mean()
SSE = d1.dot(d1)
SST = d2.dot(d2)
r_square = 1 - SSE / SST
print(r_square)


# Example
# Demonstrating Moore's Law
"""
Moore's law is that the number of transistors on integrated circuits 
doubles about every two years. 

Intel executive David House said the period was "18 months". 

He predicted that period for a doubling in chip performance: 
    a combination of the effect of more transistors and their being faster.
"""
import re
X = []
Y = []
non_decimal = re.compile(r'[^\d]+')

for line in open('moore.csv'):
    r = line.split('\t')
    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)
# Plot to see what it looks like
plt.scatter(X, Y)
plt.show()
# Plot to see what it looks like
Y = np.log(Y)
plt.scatter(X, Y)
plt.show()
# Apply the equations we learned to calculate a and b
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator
# Calculated predicted Y
Yhat = a * X + b
# Plot it all
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()
# Calculating r-square
d1 = Y - Yhat
d2 = Y - Y.mean()
SSE = d1.dot(d1)
SST = d2.dot(d2)
r_square = 1 - SSE / SST
print(r_square)
# Some Mathematics
"""
log(tc) = a * year + b

tc = exp(b) * exp(a * year)

2 * tc = 2 * exp(b) * exp(a * year) = exp(ln(2)) * exp(b) * exp(a * year)
       = exp(b) * exp(a * year + ln(2))

exp(b) * exp(a * year2) = exp(b) * exp(a * year1 + ln2)

a * year2 = a * year1 + ln2

year2 = year1 + (ln2 / a)
"""
print('time to double:', np.log(2) / a, 'years')
