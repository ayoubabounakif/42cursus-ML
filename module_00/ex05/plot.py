import numpy as np
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader

prediction = SourceFileLoader("prediction", "../ex04/prediction.py").load_module()

def plot(x, y, theta):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray): return None
    if len(x) == 0 or len(y) == 0 or len(theta) == 0: return None
    if len(x.shape) != 1 or len(y.shape) != 1 or len(theta.shape) != 2: return None
    if x.shape[0] != y.shape[0] or theta.shape[0] != 2 or theta.shape[1] != 1: return None
    y_hat = prediction.predict_(x, theta)
    plt.scatter(x, y)
    plt.plot(x, y_hat, color='red')
    plt.show()

def main():
    x = np.arange(1,6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])

    theta1 = np.array([[4.5],[-0.2]])

    plot(x, y, theta1)

    theta2 = np.array([[-1.5],[2]])
    plot(x, y, theta2)

    theta3 = np.array([[3],[0.3]])
    plot(x, y, theta3)



if __name__ == '__main__':
    main()