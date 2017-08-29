import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("hw4data1.txt", header=None)

X = data[[0,1]].values
y = data[2].values

pass_x = X[y == 1]
fail_x = X[y == 0]

plt.figure(figsize=(8,6))
plt.scatter(pass_x[:,0], pass_x[:,1], color='green', marker='o', label='pass')
plt.scatter(fail_x[:,0], fail_x[:,1], color='red', marker='o', label='fail')
plt.legend(loc='best')

plt.title('Pass vs Fail')
plt.xlabel('exam 1')
plt.ylabel('exam 2')

plt.show()

def append_bias(x):
    return np.append(np.ones((x.shape[0],1)), x, axis=1)

def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def loss_function(X, y, theta, delta):
    predictions = sigmoid(np.dot(X, theta))
    loss = -1 * (np.dot(y.T, np.log(predictions)) + np.dot(1-y.T,np.log(1-predictions))) + delta**2 * np.dot(theta.T, theta)
    return loss

# append 1s to X
X_train = np.append(np.ones((X.shape[0],1)), X, axis=1)

def gradient_descent(X, y, epochs, learning_rate, delta, enable_print_loss = False):
    theta = np.matrix(np.zeros(X.shape[1])).T
    losses = []
    for step in range(epochs):
        predictions = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, y - predictions)
        gradient = gradient + ((delta ** 2) / (X.shape[0])) * theta
        theta = theta + learning_rate * gradient
        loss = loss_function(X, y, theta,delta)[0,0]
        losses.append(loss)
        if enable_print_loss:
            print(loss)
    return theta, loss,losses

X_train = np.matrix(X_train)
X_train.shape
y_train = np.matrix(y).T
y_train.shape

gradient_descent(X_train,y_train, 100, 1.0, 0, True)

X_train_norm = np.matrix(X_train)
X_train_norm.shape

std = np.std(X_train, axis=0)
mean = np.mean(X_train, axis=0)
print ("mean {}" , mean[0])
print ("std {}", std[0])


X_train_norm = np.matrix(X_train)
X_train_norm[:,1] = (X_train_norm[:,1] - mean[0,1])/std[0,1]
X_train_norm[:,2] = (X_train_norm[:,2] - mean[0,2])/ std[0,2]
X_train_norm


def gradient_and_calc_accuracy(X, y, epochs, learning_rate, delta):
    print("Using epochs:{0}, learning_rate:{1}, delta:{2}".format(epochs, learning_rate,delta))
    theta, loss, losses = gradient_descent(X,y, epochs, learning_rate, delta)
    preds = np.round(sigmoid(np.dot(X, theta)))
    print('Accuracy for: {0}'.format((preds == y).sum().astype(float) / len(preds)))
    print("loss: {0}".format(loss))
    return theta, losses


theta_1, losses_1 = gradient_and_calc_accuracy(X_train_norm,y_train,100,1.0,0)
theta_001, losses_001 = gradient_and_calc_accuracy(X_train_norm,y_train,100,0.01,0)
theta_01,losses_01 = gradient_and_calc_accuracy(X_train_norm,y_train,100,0.1,0)
theta_10,losses_10 = gradient_and_calc_accuracy(X_train_norm,y_train,100,10,0)
theta_100,losses_100 = gradient_and_calc_accuracy(X_train_norm,y_train,100,100,0)


plt.plot(losses_001)
plt.plot(losses_01)
plt.plot(losses_1)
plt.plot(losses_10)
plt.plot(losses_100)

plt.show()


def draw_2D_contour(theta,bb,map_features=None, levels_contour = [0.5]):
    xx, yy = np.meshgrid(np.arange(bb[0],bb[1],0.01),np.arange(bb[2],bb[3],0.01))
    input_features = np.vstack((xx.flatten(),yy.flatten())).T
    if( map_features is None):
        mapped_features = append_bias(input_features)
    else:
        mapped_features = append_bias(map_features(input_features))
    zz = sigmoid(np.dot(mapped_features,theta))
    zz = np.reshape(zz,xx.shape)
    CS = plt.contour(xx, yy, zz, levels= levels_contour)
    plt.clabel(CS, inline=1, fontsize=10)


def draw_2D_contour_auto(theta,bb,map_features=None):
    xx, yy = np.meshgrid(np.arange(bb[0],bb[1],0.01),np.arange(bb[2],bb[3],0.01))
    input_features = np.vstack((xx.flatten(),yy.flatten())).T
    if( map_features is None):
        mapped_features = append_bias(input_features)
    else:
        mapped_features = append_bias(map_features(input_features))
    zz = sigmoid(np.dot(np.matrix(mapped_features), theta))
    zz = np.reshape(zz,xx.shape)
    CS = plt.contour(xx, yy, zz)
    plt.clabel(CS, inline=1, fontsize=10)


draw_2D_contour_auto(theta_01, [-2,2,-2,2])

pass_x = X_train_norm[y == 1]
fail_x = X_train_norm[y == 0]
plt.scatter(np.array(pass_x[:,1]), np.array(pass_x[:,2]), color='green', marker='o', label='pass')
plt.scatter(np.array(fail_x[:,1]), np.array(fail_x[:,2]), color='red', marker='o', label='fail')


X_test = np.matrix([[45, 85], [60,60],[90,30],[80,50]])
X_test = append_bias(X_test)
mean_test = np.mean(X_test, axis=0)
std_test = np.std(X_test, axis=0)

mean_test
std_test
X_test[:,1] = (X_test[:,1] - mean[0,1])/ std[0,1]
X_test[:,2] = (X_test[:,2] - mean[0,2])/ std[0,2]
X_test


sigmoid(np.dot(X_test, theta_01))
predictions = np.round(sigmoid(np.dot(X_test, theta_01)))
predictions


data_2 = pd.read_csv("hw4data2.txt", header=None)

X_2 = data_2[[0,1]].values
y_2 = data_2[2].values

ones = X_2[y_2 == 1]
zeroes = X_2[y_2 == 0]

plt.figure(figsize=(8,6))
plt.scatter(ones[:,0], ones[:,1], color='green', marker='o', label='ones')
plt.scatter(zeroes[:,0], zeroes[:,1], color='red', marker='o', label='zeroes')
plt.legend(loc='best')

plt.title('ones / zeroes')
plt.xlabel('feature 1')
plt.ylabel('feature 2')

def map_features(x, power=6):
    m = x
    for i in range (2, power + 1):
        for j in range(0, i + 1):
            exponent = i - j
            new_column  = np.matrix(np.multiply(np.power(m[:,0], exponent), np.power(m[:,1], j)))
            new_column = new_column.reshape(x.shape[0],1)
            m = np.concatenate([m, new_column], axis=1)
    return m


m = map_features(X_2, 6)
m.shape

mean_2 = np.mean(m, axis = 0)
std_2 = np.std(m, axis=0)
mean_2
std_2

def normalize(x, mean, std):
    normalized = x
    n = x.shape[1]
    for i in range(0,n):
        x[:,i] = (x[:,i] - mean[0,i])/ std[0,i]
    return normalized

a = normalize(m, mean_2, std_2)
a.shape

a_test = append_bias(a)

y_train = np.matrix(y_2).T
y_train.shape

theta_1, losses_1 = gradient_and_calc_accuracy(a_test,y_train,1000,1.0,0)
theta_001, losses_001 = gradient_and_calc_accuracy(a_test,y_train,1000,0.01,0)
theta_01,losses_01 = gradient_and_calc_accuracy(a_test,y_train,1000,0.1,0)
theta_10,losses_10 = gradient_and_calc_accuracy(a_test,y_train,1000,10,0)
theta_100,losses_100 =gradient_and_calc_accuracy(a_test,y_train,1000,100,0)


draw_2D_contour_auto(theta_001, [-1.5,1.5,-1.5,1.5], map_features=map_features)

plt.scatter(ones[:,0], ones[:,1], color='green', marker='o', label='ones')
plt.scatter(zeroes[:,0], zeroes[:,1], color='red', marker='o', label='zeroes')
plt.legend(loc='best')

plt.title('ones / zeroes')
plt.xlabel('feature 1')
plt.ylabel('feature 2')

plt.plot(losses_001)
plt.plot(losses_01)
plt.plot(losses_1)
plt.plot(losses_10)
plt.plot(losses_100)

theta_1, losses_1 = gradient_and_calc_accuracy(a_test,y_train,1000,1.0,10**0.5)
theta_001, losses_001 = gradient_and_calc_accuracy(a_test,y_train,1000,0.01,10**0.5)
theta_01,losses_01 = gradient_and_calc_accuracy(a_test,y_train,1000,0.1,10**0.5)
theta_10,losses_10 = gradient_and_calc_accuracy(a_test,y_train,1000,10,10**0.5)
theta_100,losses_100 =gradient_and_calc_accuracy(a_test,y_train,1000,100,10**0.5)


draw_2D_contour_auto(theta_001, [-1.5,1.5,-1.5,1.5], map_features=map_features)

plt.scatter(ones[:,0], ones[:,1], color='green', marker='o', label='ones')
plt.scatter(zeroes[:,0], zeroes[:,1], color='red', marker='o', label='zeroes')
plt.legend(loc='best')

plt.title('ones / zeroes')
plt.xlabel('feature 1')
plt.ylabel('feature 2')

plt.plot(losses_001)
plt.plot(losses_01)
plt.plot(losses_1)
plt.plot(losses_10)
plt.plot(losses_100)

theta_1, losses_1 = gradient_and_calc_accuracy(a_test,y_train,1000,1.0,1**0.5)
theta_001, losses_001 = gradient_and_calc_accuracy(a_test,y_train,1000,0.01,1**0.5)
theta_01,losses_01 = gradient_and_calc_accuracy(a_test,y_train,1000,0.1,1**0.5)
theta_10,losses_10 = gradient_and_calc_accuracy(a_test,y_train,1000,10,1**0.5)
theta_100,losses_100 =gradient_and_calc_accuracy(a_test,y_train,1000,100,1**0.5)

draw_2D_contour_auto(theta_001, [-1.5,1.5,-1.5,1.5], map_features=map_features)

plt.scatter(ones[:,0], ones[:,1], color='green', marker='o', label='ones')
plt.scatter(zeroes[:,0], zeroes[:,1], color='red', marker='o', label='zeroes')
plt.legend(loc='best')

plt.title('ones / zeroes')
plt.xlabel('feature 1')
plt.ylabel('feature 2')

plt.plot(losses_001)
plt.plot(losses_01)
plt.plot(losses_1)
plt.plot(losses_10)
plt.plot(losses_100)