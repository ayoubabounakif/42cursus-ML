import numpy as np
from importlib.machinery import SourceFileLoader

tools = SourceFileLoader("tools", "../ex03/tools.py").load_module()

def predict_(x, theta):
    if len(x) == 0 or len(theta) == 0: return None
    if len(x.shape) != 1 or len(theta.shape) != 2: return None
    x_ = tools.add_intercept(x)
    y_hat = np.dot(x_, theta)
    return y_hat
    

def main():
    x = np.arange(1,6)

    theta1 = np.array([[5], [0]])
    print(predict_(x, theta1)) # array([[5.], [5.], [5.], [5.], [5.]])

    theta2 = np.array([[0], [1]])
    print(predict_(x, theta2)) # array([[1.], [2.], [3.], [4.], [5.]])

    theta3 = np.array([[5], [3]])
    print(predict_(x, theta3)) # array([[ 8.], [11.], [14.], [17.], [20.]])

    theta4 = np.array([[-3], [1]])
    print(predict_(x, theta4)) # array([[-2.], [-1.], [ 0.], [ 1.], [ 2.]])

if __name__ == '__main__':
    main()
