import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
MyLR = SourceFileLoader("mylinearregression", "../ex05/mylinearregression.py").load_module().MyLinearRegression

def lr_age(x, y):
    myLR_age = MyLR(thetas = np.array([[1000.0], [-1.0]]), alpha = 2.5e-5, max_iter = 100000)
    myLR_age.fit_(x, y)
    y_hat = myLR_age.predict_(x)
    loss = myLR_age.mse_(y_hat, y)
    print('---- Part one ----')
    print(f"bias: {myLR_age.thetas[0]}, weight: {myLR_age.thetas[1]}")
    print(f"Loss: {loss}")
    print('------------------')
    plt.plot(x, y, 'o', color='darkblue')
    plt.plot(x, y_hat, 'o', color='cornflowerblue', markersize=3)
    plt.grid()
    plt.legend(['Sell price', 'Predicted sell price'], loc='lower left')
    plt.xlabel(f"x\u2081: age (in years)")
    plt.ylabel(f"y: sell price (in keuros)")
    plt.show()

def lr_thrust(x, y):
    myLR_thrust = MyLR(thetas = np.array([[60.0], [3.0]]), alpha = 2.5e-5, max_iter = 150000)
    myLR_thrust.fit_(x, y)
    y_hat = myLR_thrust.predict_(x)
    loss = myLR_thrust.mse_(y_hat, y)
    print('---- Part two ----')
    print(f"bias: {myLR_thrust.thetas[0]}, weight: {myLR_thrust.thetas[1]}")
    print(f"Loss: {loss}")
    print('------------------')
    plt.plot(x, y, 'o', color='#77AC30')
    plt.plot(x, y_hat, 'o', color='#00FF00', markersize=3)
    plt.grid()
    plt.legend(['Sell price', 'Predicted sell price'], loc='upper left')
    plt.xlabel(f"x\u2082: thurst power (in 10Km/s)")
    plt.ylabel(f"y: sell price (in keuros)")
    plt.show()

def lr_distance(x, y):
    myLR_distance = MyLR(thetas = np.array([[700.0], [-1.0]]), alpha = 2.5e-5, max_iter = 150000)
    myLR_distance.fit_(x, y)
    y_hat = myLR_distance.predict_(x)
    loss = myLR_distance.mse_(y_hat, y)
    print('---- Part two ----')
    print(f"bias: {myLR_distance.thetas[0]}, weight: {myLR_distance.thetas[1]}")
    print(f"Loss: {loss}")
    print('------------------')
    plt.plot(x, y, 'o', color='purple')
    plt.plot(x, y_hat, 'o', color='#EE82EE', markersize=3)
    plt.grid()
    plt.legend(['Sell price', 'Predicted sell price'], loc='upper right')
    plt.xlabel(f"x\u2083: distance totalizer of spacecraft (in Tmeters)")
    plt.ylabel(f"y: sell price (in keuros)")
    plt.show()

def main():
    y = pd.read_csv('./spacecraft_data.csv', usecols=['Sell_price']).values
    X = pd.read_csv('./spacecraft_data.csv', usecols=['Age', 'Thrust_power', 'Terameters']).values

    x = X[:, 0].reshape(-1, 1)
    lr_age(x, y)

    x = X[:, 1].reshape(-1, 1)
    lr_thrust(x, y)

    x = X[:, 2].reshape(-1, 1)
    lr_distance(x, y)


if __name__ == '__main__':
    main()