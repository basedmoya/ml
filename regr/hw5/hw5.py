import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_svmlight_file


# Homework 5
# Problem 1
# In this problem you will implement gradient descent and its variants to optimize a ridge regression model.

# functions
def append_bias(x):
    return np.append(np.ones((x.shape[0], 1)), x, axis=1)


def sigmoid(x):
    return 1 /(1 + np.exp(-x))


def ridge_regression(x, y, scalar):
    dimension = x.transpose().shape[0]
    scalar_identity = (scalar ** 2) * np.identity(dimension)
    return np.dot(np.dot(np.linalg.inv(np.add(np.dot(x.transpose(), x), scalar_identity)), x.transpose()), y)


def mse(y, y_hat):
    n = y.shape[0]
    return sse(y, y_hat)/n


def sse(y, y_hat):
    e = y - y_hat
    ss = np.dot(e.T, e)
    return ss


def batch_gradient_descent(x, y, epochs, learning_rate, delta):
    theta = np.matrix(np.zeros(x.shape[1])).T
    for step in range(epochs):
        predictions = np.dot(x, theta)
        gradient = np.dot(x.T, y - predictions) - (delta ** 2) * theta
        theta = theta + (learning_rate / x.shape[0]) * gradient

    return theta


def sgd(x, y, epochs, learning_rate, delta):
    theta = np.matrix(np.zeros(x.shape[1])).T
    losses = []
    for step in range(epochs):
        for i, xi in enumerate(x):
            predictions = np.dot(xi, theta)
            gradient = np.dot(xi.T, y[i, :] - predictions) - (1.0 / x.shape[0]) * (delta ** 2) * theta
            theta = theta + learning_rate * gradient
            losses.append(sse(y_train, np.dot(x, theta))[0, 0])

    return theta, losses


# Read main data set, and transform them to matrices
X_train, y_train = load_svmlight_file("hw5_train.txt")
X_train = np.matrix(X_train.toarray())
y_train = np.matrix(y_train).T

X_train = append_bias(X_train)

X_test, y_test = load_svmlight_file("hw5_test.txt")
X_test = np.matrix(X_test.toarray())
y_test = np.matrix(y_test).T

X_test = append_bias(X_test)

# Question 1:
# Fit a ridge regression model using the provided training dataset. Don’t forget to include the bias column.
# Follow the steps below to implement stochastic gradient descent, mini-batch gradient descent,
# and batch gradient descent. Please note the following implementation details.

# Step 1:
# Learn a ridge regression model using the closed-form solution. Set δ2 = 1. Report the training and test error
theta_ridge = ridge_regression(X_train, y_train, 1)

print("theta ridge")
print(theta_ridge)


# train error
y_hat = np.dot(X_train, theta_ridge)
train_total_error = sse(y_train, y_hat)
train_mean_error = mse(y_train, y_hat)
print("sum of squared errors in y_train was {}".format(train_total_error))

# test error
y_hat = np.dot(X_test, theta_ridge)

test_total_error = sse(y_test, y_hat)
test_mean_error = mse(y_test, y_hat)
print("sum of squared errors in y_test was {}".format(test_total_error))

# Step 2:
# Learn a ridge regression model using batch gradient descent with fixed learning rate. Set δ2 = 1.
#  For the learning rate η0, try values 0.01, 0.1, and 1 and pick the one that gives you a training and test error
# that are close to the errors you got from step 1. Run 100 epochs of training.

# using learning rate = 0.01
theta_bgd = batch_gradient_descent(X_train, y_train, 100, 0.01, 1)


print("\ntheta bgd for 0.01")
print(theta_bgd)

# train error
y_hat = np.dot(X_train, theta_bgd)
train_total_error = sse(y_train, y_hat)
train_mean_error = mse(y_train, y_hat)
print("sum of squared errors in y_train was {}".format(train_total_error))

# test error
y_hat = np.dot(X_test, theta_bgd)

test_total_error = sse(y_test, y_hat)
test_mean_error = mse(y_test, y_hat)
print("sum of squared errors in y_test was {}".format(test_total_error))


# using learning rate = 0.1
theta_bgd = batch_gradient_descent(X_train, y_train, 100, 0.1, 1)


print("\ntheta bgd for 0.1")
print(theta_bgd)

# train error
y_hat = np.dot(X_train, theta_bgd)
train_total_error = sse(y_train, y_hat)
train_mean_error = mse(y_train, y_hat)
print("sum of squared errors in y_train was {}".format(train_total_error))

# test error
y_hat = np.dot(X_test, theta_bgd)

test_total_error = sse(y_test, y_hat)
test_mean_error = mse(y_test, y_hat)
print("sum of squared errors in y_test was {}".format(test_total_error))

# using learning rate = 1.0
theta_bgd = batch_gradient_descent(X_train, y_train, 100, 1.0, 1)


print("\ntheta bgd for 1.0")
print(theta_bgd)

# train error
y_hat = np.dot(X_train, theta_bgd)
train_total_error = sse(y_train, y_hat)
train_mean_error = mse(y_train, y_hat)
print("sum of squared errors in y_train was {}".format(train_total_error))

# test error
y_hat = np.dot(X_test, theta_bgd)

test_total_error = sse(y_test, y_hat)
test_mean_error = mse(y_test, y_hat)
print("sum of squared errors in y_test was {}".format(test_total_error))


print("\ntheta bgd for 1.0 was the best result")


# Step 3:
# Learn a ridge regression model using stochastic gradient descent with fixed learning rate
# using learning rate = 0.01
theta_sgd, losses_sgd = sgd(X_train, y_train, 100, 0.01, 1)


print("\ntheta sgd for 0.01")
print(theta_sgd)

# train error
y_hat = np.dot(X_train, theta_sgd)
train_total_error = sse(y_train, y_hat)
train_mean_error = mse(y_train, y_hat)
print("sum of squared errors in y_train was {}".format(train_total_error))


# test error
y_hat = np.dot(X_test, theta_sgd)

test_total_error = sse(y_test, y_hat)
test_mean_error = mse(y_test, y_hat)
print("sum of squared errors in y_test was {}".format(test_total_error))


# using learning rate = 0.1
theta_sgd, losses_sgd = sgd(X_train, y_train, 100, 0.1, 1)


print("\ntheta sgd for 0.1")
print(theta_sgd)

# train error
y_hat = np.dot(X_train, theta_sgd)
train_total_error = sse(y_train, y_hat)
train_mean_error = mse(y_train, y_hat)
print("sum of squared errors in y_train was {}".format(train_total_error))


# test error
y_hat = np.dot(X_test, theta_sgd)

test_total_error = sse(y_test, y_hat)
test_mean_error = mse(y_test, y_hat)
print("sum of squared errors in y_test was {}".format(test_total_error))

# using learning rate = 1.0
print("\nnot able to calculate with learning rate 1.0")

print("\nbest results for theta sgd were for learning rate 0.01")


# Step 4:
# Denote by p the best learning rate you picked in step 3.
# Run stochastic gradient descent again with two different fixed learning rates: p∗0.1 and p∗10
learning_rate = 0.01
theta_sgd_p_01, losses_p_01 = sgd(X_train, y_train, 5, learning_rate * 0.1, 1)
theta_sgd_p_1, losses_p_1 = sgd(X_train, y_train, 5, learning_rate, 1)
theta_sgd_p_10, losses_p_10 = sgd(X_train, y_train, 5, learning_rate * 10, 1)


plt.plot(np.arange(len(losses_p_01)),np.array(losses_p_01), label='p*0.1')
plt.plot(np.arange(len(losses_p_1)),np.array(losses_p_1), label='p')
plt.plot(np.arange(len(losses_p_10)),np.array(losses_p_10), label='p*10')

plt.title('SGD by learning rate')
plt.xlabel('# iterations')
plt.ylabel('training errors')
plt.legend(loc='best')
plt.show()






