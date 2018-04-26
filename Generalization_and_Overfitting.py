import numpy as np
import matplotlib.pyplot as plt

# Make some fake data
N = 100
X = np.linspace(0, 6*np.pi, N)
Y = np.sin(X)  # the range of the elements in Y is from -1 to 1

# Plot it to get a more intuition thought on the data
plt.plot(X, Y)
plt.show()

def make_poly(X, deg):
    """
    This function is to create a matrix which will have x, x^2, x^3,..., X^deg.
    """
    n = len(X)
    data = [np.ones(n)]
    for d in range(deg):
        data.append(X ** (d + 1))
    return np.vstack(data).T  # It would be better to understand if running the code seperately.

def fit(X, Y):
    """
    Return the weights(parameters)."""
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))  


def fit_and_display(X, Y, sample, deg):
    N = len(X)
    train_idx = np.random.choice(N, sample)  # Sample should be an integer, not ratio.
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    plt.scatter(Xtrain, Ytrain)
    plt.show()

    # fit polynomial
    Xtrain_poly = make_poly(Xtrain, deg)
    w = fit(Xtrain_poly, Ytrain)

    # display the polynomial
    X_poly = make_poly(X, deg)
    Y_hat = X_poly.dot(w)
    plt.plot(X, Y)
    plt.plot(X, Y_hat)
    plt.scatter(Xtrain, Ytrain)
    plt.title("deg = %d" % deg)
    plt.show()

# Let's try to show what we've done until now
for deg in (5, 6, 7, 8, 9):
    fit_and_display(X, Y, 10, deg)


def get_mse(Y, Yhat):
    d = Y - Yhat
    return d.dot(d) / len(d)


def plot_train_vs_test_curves(X, Y, sample=20, max_deg=20):
    N = len(X)
    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    test_idx = [idx for idx in range(N) if idx not in train_idx]
    # test_idx = np.random.choice(N, sample)
    Xtest = X[test_idx]
    Ytest = Y[test_idx]

    mse_trains = []
    mse_tests = []
    for deg in range(max_deg+1):
        Xtrain_poly = make_poly(Xtrain, deg)
        w = fit(Xtrain_poly, Ytrain)
        Yhat_train = Xtrain_poly.dot(w)
        mse_train = get_mse(Ytrain, Yhat_train)

        Xtest_poly = make_poly(Xtest, deg)
        Yhat_test = Xtest_poly.dot(w)
        mse_test = get_mse(Ytest, Yhat_test)

        mse_trains.append(mse_train)
        mse_tests.append(mse_test)

    plt.plot(mse_trains, label="train mse")
    plt.plot(mse_tests, label="test mse")
    plt.legend()
    plt.show()

    plt.plot(mse_trains, label="train mse")
    plt.legend()
    plt.show()

# Show how it works!
plot_train_vs_test_curves(X, Y)
