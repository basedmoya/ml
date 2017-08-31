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

def closed_form(X_train, y_train, delta):

    ## Closed form Ridge Regression solution
    theta_ridge = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train) + delta**2 * np.identity(d_train + 1)), X_train.T), y_train).reshape(d_train + 1,1)
    return  theta_ridge

def predict(X_train, theta):
    n = X_train.shape[0]
    return np.dot(X_train, theta).reshape(n,1)

def print_errors(y, y_hat):
    train_total_error = sse(y, y_hat)
    train_mean_error = mse(y, y_hat)
    print("sum of squared errors in y_train was {}".format(train_total_error))
    # print("mean of squared errors in y_train was {}".format(train_mean_error))

print("Closed form results")
theta_ridge = closed_form(X_train, y_train, delta)
y_hat = predict(X_train, theta_ridge)
print_errors(y_train, y_hat)

y_hat = predict(X_test, theta_ridge)
print_errors(y_test, y_hat)


## Fixed Learn rate, batch gradient descent
delta = 1
# try values 0.01, 0.1, 1
learning_rate = 1.0
epochs = 100

def batch_gradient_descent(X, y, delta, learning_rate, epochs):
    losses = []
    # initialize theta
    theta_bgd = np.zeros((d_train + 1, 1))
    for iteration in range(epochs):
        residuals = y - np.dot(X, theta_bgd).reshape(n_train, 1)
        gradient = np.dot(X.T, residuals) - delta**2 * theta_bgd
        # print("gradient is {}".format(gradient))

        ## NOTE: this is where the batch happens. i.e divide learning rate by n
        theta_bgd = theta_bgd + learning_rate/n_train * gradient
        total_loss = sse(y, np.dot(X, theta_bgd).reshape(n_train, 1))
        losses.append(total_loss)
        # print("total loss is {}".format(sse(y_train, np.dot(X_train, theta_bgd).reshape(n_train, 1))))
        # print(theta_bgd)
    return theta_bgd, losses

print("Batch gradient results")
theta_bgd, losses_bgd_1 = batch_gradient_descent(X_train, y_train, delta, learning_rate, epochs)
y_hat = predict(X_train, theta_bgd)
print_errors(y_train, y_hat)

y_hat = predict(X_test, theta_bgd)
print_errors(y_test, y_hat)

## learning_rate = 1 gave the best results
print("fixed learning rate was {}, recorded best was 1.0".format(learning_rate))


## Fixed learning rate, stochastic gradient descent
delta = 1
# try values 0.01, 0.1, 1
learning_rate = 0.01
epochs = 100


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

print("Stochastic gradient results")
theta_sgd, losses_sgd = stochastic_gradient_descent(X_train, y_train, learning_rate, epochs, delta)
y_hat = predict(X_train, theta_sgd)
print_errors(y_train, y_hat)
y_hat = predict(X_test, theta_sgd)
print_errors(y_test, y_hat)
print("fixed learning rate was {}, recorded best was 0.01".format(learning_rate))


## Step 4: p  * learning rate, stochastic gradient descent
delta = 1
# try values 0.01, 0.1, 1
p = 0.01
epochs = 5


theta_sgd_01, losses_01 = stochastic_gradient_descent(X_train, y_train, learning_rate=0.1 * p, epochs=5, delta=1)
theta_sgd_1, losses_1 = stochastic_gradient_descent(X_train, y_train, learning_rate=1.0 * p, epochs=5, delta=1)
theta_sgd_10, losses_10 = stochastic_gradient_descent(X_train, y_train, learning_rate=10.0 * p, epochs=5, delta=1)

# plt.plot(np.arange(len(losses_01)),np.array(losses_01))
# plt.plot(np.arange(len(losses_1)),np.array(losses_1))
# plt.plot(np.arange(len(losses_10)),np.array(losses_10))
# plt.show()

#
# print_results(X_train, theta_sgd_01)
# print_results(X_train, theta_sgd_1)
# print_results(X_train, theta_sgd_10)


## Step 5: mini batch

## Use inverse scaling heuristic
batch_size = 10
epochs = 100
delta = 1
learning_rate = 0.01

def smartLearningRateFn(eta_0, t):
    if t==0:
        return eta_0
    power_t = 0.25
    return eta_0*1.0/(t**power_t)


def batch_gradient_descent_ish(X, y, delta, eta_0, epochs):
    losses = []
    d_train = X.shape[1]
    n_train = X.shape[0]
    # initialize theta
    theta_bgd = np.zeros((d_train, 1))
    for iteration in range(epochs):
        residuals = y - np.dot(X, theta_bgd).reshape(n_train, 1)
        gradient = np.dot(X.T, residuals) - delta**2 * theta_bgd
        # print("gradient is {}".format(gradient))

        iterations = 1+iteration
        learning_rate = smartLearningRateFn(eta_0, iterations)
        # print(iterations)
        # print(learning_rate)
        ## NOTE: this is where the batch happens. i.e divide learning rate by n
        theta_bgd = theta_bgd + learning_rate/n_train * gradient
        total_loss = sse(y, np.dot(X, theta_bgd).reshape(n_train, 1))
        losses.append(total_loss)
        # print("total loss is {}".format(sse(y_train, np.dot(X_train, theta_bgd).reshape(n_train, 1))))
        # print(theta_bgd)
    return theta_bgd, losses

def stochastic_gradient_descent_ish(X_train, y_train, eta_0, epochs, delta):
    loss_progress = []

    d_train = X_train.shape[1]
    n_train = X_train.shape[0]
    # initialize theta
    theta_sgd = np.zeros((d_train, 1))
    for iteration in range(epochs):

        for i, xi in enumerate(X_train):
            xi = xi.reshape(1,7)
            assert(xi.shape == (1,7))

            residuals = y_train[i,:] - np.dot(xi, theta_sgd).reshape(1, 1)
            ## NOTE: this is where the stochastic happens. i.e divide theta term rate by n
            gradient = np.dot(xi.T, residuals) - 1.0/n_train * delta**2 * theta_sgd
            # print("gradient is {}".format(gradient))

            iterations = i+1 + X_train.shape[0]*iteration
            learning_rate = smartLearningRateFn(eta_0, iterations)
            theta_sgd = theta_sgd + learning_rate * gradient
            loss_progress.append(sse(y_train, predict(X_train, theta_sgd)))
            # print("total loss is {}".format(sse(y_train, np.dot(X_train, theta_sgd).reshape(n_train, 1))))
            # print(theta_sgd)
    return theta_sgd, loss_progress

def mini_batch_gradient_descent_ish(X, y, eta_0, epochs, delta, batch_size):
    losses = []

    d_train = X.shape[1]
    n_train = X.shape[0]
    # initialize theta
    theta = np.zeros((d_train, 1))
    for iteration in range(epochs):

        # Since this is batch, we should proceed through the set m samples at a time
        for i in range(0,X.shape[0],batch_size):
            xm = X[i:i+batch_size,:]
            ym = y[i:i+batch_size,:]
            # assert(xm.shape==(batch_size, 7))

            residuals = ym - predict(xm, theta)
            gradient = np.dot(xm.T, residuals) - batch_size * 1.0/n_train * delta ** 2 * theta
            # print("gradient is {}".format(gradient))

            iterations = i+1 + X.shape[0]*iteration
            learning_rate = smartLearningRateFn(eta_0, iterations)

            # print(iterations)
            ## NOTE: this is where the batch happens. i.e divide learning rate by n
            theta = theta + learning_rate / batch_size * gradient
            total_loss = sse(y, predict(X, theta))
            losses.append(total_loss)
    return theta, losses

print("Mini Batch gradient results 0.01")
eta_0 = 0.01
theta_mbgd, losses_mbgd = mini_batch_gradient_descent_ish(X_train, y_train, eta_0, epochs, delta, batch_size)
y_hat = predict(X_train, theta_mbgd)
print_errors(y_train, y_hat)
y_hat = predict(X_test, theta_mbgd)
print_errors(y_test, y_hat)
print("Mini Batch gradient results 0.1")
eta_0 = 0.1
theta_mbgd, losses_mbgd = mini_batch_gradient_descent_ish(X_train, y_train, eta_0, epochs, delta, batch_size)
y_hat = predict(X_train, theta_mbgd)
print_errors(y_train, y_hat)
y_hat = predict(X_test, theta_mbgd)
print_errors(y_test, y_hat)
print("Mini Batch gradient results 1.0")
eta_0 = 1.0
theta_mbgd, losses_mbgd = mini_batch_gradient_descent_ish(X_train, y_train, eta_0, epochs, delta, batch_size)
y_hat = predict(X_train, theta_mbgd)
print_errors(y_train, y_hat)
y_hat = predict(X_test, theta_mbgd)
print_errors(y_test, y_hat)



print(" Batch gradient results 0.01")
eta_0 = 0.01
theta_mbgd, losses_mbgd = batch_gradient_descent_ish(X_train, y_train, eta_0=eta_0, epochs=epochs, delta=delta)
y_hat = predict(X_train, theta_mbgd)
print_errors(y_train, y_hat)
y_hat = predict(X_test, theta_mbgd)
print_errors(y_test, y_hat)
print(" Batch gradient results 0.1")
eta_0 = 0.1
theta_mbgd, losses_mbgd = batch_gradient_descent_ish(X_train, y_train, eta_0=eta_0, epochs=epochs, delta=delta)
y_hat = predict(X_train, theta_mbgd)
print_errors(y_train, y_hat)
y_hat = predict(X_test, theta_mbgd)
print_errors(y_test, y_hat)
print(" Batch gradient results 1.0")
eta_0 = 1.0
theta_mbgd, losses_mbgd = batch_gradient_descent_ish(X_train, y_train, eta_0=eta_0, epochs=epochs, delta=delta)
y_hat = predict(X_train, theta_mbgd)
print_errors(y_train, y_hat)
y_hat = predict(X_test, theta_mbgd)
print_errors(y_test, y_hat)

print(" Stochastic gradient results 0.01")
eta_0 = 0.01
theta_mbgd, losses_mbgd = stochastic_gradient_descent_ish(X_train, y_train, eta_0=eta_0, epochs=epochs, delta=delta)
y_hat = predict(X_train, theta_mbgd)
print_errors(y_train, y_hat)
y_hat = predict(X_test, theta_mbgd)
print_errors(y_test, y_hat)
print(" Stochastic gradient results 0.1")
eta_0 = 0.1
theta_mbgd, losses_mbgd = stochastic_gradient_descent_ish(X_train, y_train, eta_0=eta_0, epochs=epochs, delta=delta)
y_hat = predict(X_train, theta_mbgd)
print_errors(y_train, y_hat)
y_hat = predict(X_test, theta_mbgd)
print_errors(y_test, y_hat)
print(" Stochastic gradient results 1.0")
eta_0 = 1.0
theta_mbgd, losses_mbgd = stochastic_gradient_descent_ish(X_train, y_train, eta_0=eta_0, epochs=epochs, delta=delta)
y_hat = predict(X_train, theta_mbgd)
print_errors(y_train, y_hat)
y_hat = predict(X_test, theta_mbgd)
print_errors(y_test, y_hat)

eta_0_mini_batch = 0.1
eta_0_batch = 1.0
eta_0_stochastic = 0.01


## Step 6
theta_best_minibatch, losses_best_minibatch = mini_batch_gradient_descent_ish(X_train, y_train, eta_0_mini_batch, epochs, delta, batch_size)

theta_best_batch, losses_best_batch = batch_gradient_descent_ish(X_train, y_train, eta_0=eta_0_batch, epochs=epochs, delta=delta)

theta_best_stochastic, losses_best_stochastic = stochastic_gradient_descent_ish(X_train, y_train, eta_0=eta_0_stochastic, epochs=epochs, delta=delta)
a = plt.plot(np.arange(len(losses_best_minibatch)*batch_size, step=batch_size), np.array(losses_best_minibatch)[:,0])
b = plt.plot(np.arange(len(losses_best_batch)*X_train.shape[0], step=X_train.shape[0]), np.array(losses_best_batch)[:,0])
c = plt.plot(np.arange(len(losses_best_stochastic)), np.array(losses_best_stochastic)[:,0])
plt.title("Training Loss")
plt.legend(['mini batch','batch','stochastic'])
plt.show()

theta_best_minibatch, losses_best_minibatch = mini_batch_gradient_descent_ish(X_test, y_test, eta_0_mini_batch, epochs, delta, batch_size)

theta_best_batch, losses_best_batch = batch_gradient_descent_ish(X_test, y_test, eta_0=eta_0_batch, epochs=epochs, delta=delta)
theta_best_stochastic, losses_best_stochastic = stochastic_gradient_descent_ish(X_test, y_test, eta_0=eta_0_stochastic, epochs=epochs, delta=delta)
plt.figure()
a = plt.plot(np.arange(len(losses_best_minibatch)*batch_size, step=batch_size), np.array(losses_best_minibatch)[:,0] ,label='a')
b = plt.plot(np.arange(len(losses_best_batch)*X_train.shape[0], step=X_train.shape[0]), np.array(losses_best_batch)[:,0], label='b')
c = plt.plot(np.arange(len(losses_best_stochastic)), np.array(losses_best_stochastic)[:,0], label='c')
plt.title("Test Loss")
plt.legend(['mini batch','batch','stochastic'])
plt.show()