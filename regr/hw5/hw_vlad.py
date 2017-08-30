import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_svmlight_file

# X is (700,6) and y (700,1)
X_train,y_train = load_svmlight_file("hw5_train.txt")
X_train = X_train.toarray()


X_test,y_test = load_svmlight_file("hw5_test.txt")
X_test = X_test.toarray()

delta = 1

d_train = X_train.shape[1]
n_train = X_train.shape[0]


d_test = X_test.shape[1]
n_test = X_test.shape[0]

# add ones
X_train = np.append(np.ones((n_train,1)), X_train, 1)
X_test = np.append(np.ones((n_test,1)), X_test, 1)


# convert to np.matrix
y_train = y_train.reshape(n_train, 1)
y_test = y_test.reshape(n_test, 1)


def mse(y, y_hat):
    n = y.shape[0]
    return sse(y,y_hat)/n

def sse(y,y_hat):
    e = y-y_hat
    ss = np.dot(e.T,e)
    return ss

## Closed form Ridge Regression solution
theta_ridge = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train) + delta**2 * np.identity(d_train + 1)), X_train.T), y_train).reshape(d_train + 1,1)

print("Closed form results")
y_hat = np.dot(X_train, theta_ridge).reshape(n_train, 1)
train_total_error = sse(y_train, y_hat)
train_mean_error = mse(y_train, y_hat)
print("sum of squared errors in y_train was {}".format(train_total_error))
# print("mean of squared errors in y_train was {}".format(train_mean_error))


y_hat = np.dot(X_test, theta_ridge).reshape(n_test,1)
test_total_error = sse(y_test, y_hat)
test_mean_error = mse(y_test, y_hat)
print("sum of squared errors in y_test was {}".format(test_total_error))
# print("mean of squared errors in y_test was {}".format(test_mean_error))


## Fixed Learn rate, batch gradient descent
delta = 1
# try values 0.01, 0.1, 1
learning_rate = 1.0
epochs = 100

# initialize theta
theta_bgd = np.zeros((d_train + 1, 1))
for iteration in range(epochs):
    residuals = y_train - np.dot(X_train, theta_bgd).reshape(n_train, 1)
    gradient = np.dot(X_train.T, residuals) - delta**2 * theta_bgd
    # print("gradient is {}".format(gradient))

    ## NOTE: this is where the batch happens. i.e divide learning rate by n
    theta_bgd = theta_bgd + (learning_rate/n_train) * gradient
    # print("total loss is {}".format(sse(y_train, np.dot(X_train, theta_bgd).reshape(n_train, 1))))
    # print(theta_bgd)


print(theta_bgd)
print("Batch gradient results")
y_hat = np.dot(X_train, theta_bgd).reshape(n_train,1)
train_total_error = sse(y_train, y_hat)
train_mean_error = mse(y_train, y_hat)
print("sum of squared errors in y_train was {}".format(train_total_error))
# print("mean of squared errors in y_train was {}".format(train_mean_error))


y_hat = np.dot(X_test, theta_bgd).reshape(n_test,1)
test_total_error = sse(y_test, y_hat)
test_mean_error = mse(y_test, y_hat)
print("sum of squared errors in y_test was {}".format(test_total_error))
# print("mean of squared errors in y_test was {}".format(test_mean_error))

## learning_rate = 1 gave the best results
print("fixed learning rate was {}, recorded best was 1.0".format(learning_rate))


## Fixed learning rate, stochastic gradient descent
delta = 1
# try values 0.01, 0.1, 1
learning_rate = 0.01
epochs = 100

# initialize theta
theta_sgd = np.zeros((d_train + 1, 1))
for iteration in range(epochs):

    for i, xi in enumerate(X_train):
        xi = xi.reshape(1,7)
        assert(xi.shape == (1,7))

        residuals = y_train[i,:] - np.dot(xi, theta_sgd).reshape(1, 1)
        ## NOTE: this is where the stochastic happens. i.e divide theta term rate by n
        gradient = np.dot(xi.T, residuals) - 1.0/n_train * delta**2 * theta_sgd
        # print("gradient is {}".format(gradient))

        theta_sgd = theta_sgd + learning_rate * gradient
        # print("total loss is {}".format(sse(y_train, np.dot(X_train, theta_sgd).reshape(n_train, 1))))
        # print(theta_sgd)

print("Stochastic gradient results")
y_hat = np.dot(X_train, theta_sgd).reshape(n_train,1)
train_total_error = sse(y_train, y_hat)
train_mean_error = mse(y_train, y_hat)
print("sum of squared errors in y_train was {}".format(train_total_error))
# print("mean of squared errors in y_train was {}".format(train_mean_error))


y_hat = np.dot(X_test, theta_sgd).reshape(n_test,1)
test_total_error = sse(y_test, y_hat)
test_mean_error = mse(y_test, y_hat)
print("sum of squared errors in y_test was {}".format(test_total_error))
# print("mean of squared errors in y_test was {}".format(test_mean_error))

## learning_rate = 1 gave the best results
print("fixed learning rate was {}, recorded best was 0.01".format(learning_rate))


## Step 4: p  * learning rate, stochastic gradient descent
delta = 1
# try values 0.01, 0.1, 1
p = learning_rate
epochs = 5


def stochastic_gradient_descent(X_train, y_train, learning_rate, epochs, delta):
    loss_progress = []

    # initialize theta
    theta_sgd = np.zeros((d_train + 1, 1))
    for iteration in range(epochs):

        for i, xi in enumerate(X_train):
            xi = xi.reshape(1,7)
            assert(xi.shape == (1,7))

            residuals = y_train[i,:] - np.dot(xi, theta_sgd).reshape(1, 1)
            ## NOTE: this is where the stochastic happens. i.e divide theta term rate by n
            gradient = np.dot(xi.T, residuals) - 1.0/n_train * delta**2 * theta_sgd
            # print("gradient is {}".format(gradient))

            theta_sgd = theta_sgd + learning_rate * gradient
            loss_progress.append(sse(y_train, np.dot(X_train, theta_sgd).reshape(n_train, 1))[0,0])
            # print("total loss is {}".format(sse(y_train, np.dot(X_train, theta_sgd).reshape(n_train, 1))))
            # print(theta_sgd)
    return theta_sgd, loss_progress

def print_results(X_train, theta_sgd):
    y_hat = np.dot(X_train, theta_sgd).reshape(n_train, 1)
    train_total_error = sse(y_train, y_hat)
    train_mean_error = mse(y_train, y_hat)
    print("sum of squared errors in y_train was {}".format(train_total_error))
    # print("mean of squared errors in y_train was {}".format(train_mean_error))


    y_hat = np.dot(X_test, theta_sgd).reshape(n_test, 1)
    test_total_error = sse(y_test, y_hat)
    test_mean_error = mse(y_test, y_hat)
    print("sum of squared errors in y_test was {}".format(test_total_error))
    # print("mean of squared errors in y_test was {}".format(test_mean_error))


theta_sgd_01, losses_01 = stochastic_gradient_descent(X_train, y_train, learning_rate=0.1 * p, epochs=5, delta=1)
theta_sgd_1, losses_1 = stochastic_gradient_descent(X_train, y_train, learning_rate=1.0 * p, epochs=5, delta=1)
theta_sgd_10, losses_10 = stochastic_gradient_descent(X_train, y_train, learning_rate=10.0 * p, epochs=5, delta=1)

plt.plot(np.arange(len(losses_01)),np.array(losses_01))
plt.plot(np.arange(len(losses_1)),np.array(losses_1))
plt.plot(np.arange(len(losses_10)),np.array(losses_10))
plt.show()

#
# print_results(X_train, theta_sgd_01)
# print_results(X_train, theta_sgd_1)
# print_results(X_train, theta_sgd_10)
