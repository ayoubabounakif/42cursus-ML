import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
MyLR = SourceFileLoader("mylinearregression", "../ex05/mylinearregression.py").load_module().MyLinearRegression

def part_one(x, y):
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



def main():
    y = pd.read_csv('./spacecraft_data.csv', usecols=['Sell_price']).values
    X = pd.read_csv('./spacecraft_data.csv', usecols=['Age', 'Thrust_power', 'Terameters']).values

    x = X[:, 0].reshape(-1, 1)
    part_one(x, y)


if __name__ == '__main__':
    main()