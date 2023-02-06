import numpy as np
from importlib.machinery import SourceFileLoader

predict_ = SourceFileLoader("predict_", "../ex04/prediction.py").load_module().predict_

# Computes the squared differences between the predicted output and the expected output.
def loss_elem_(y, y_hat):
    if (isinstance(y, np.ndarray) == False or isinstance(y_hat, np.ndarray) == False): return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape: return None
    return (y_hat - y) ** 2

# Computes the MSE
def loss_(y, y_hat):
    if (isinstance(y, np.ndarray) == False or isinstance(y_hat, np.ndarray) == False): return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape: return None
    return np.sum(loss_elem_(y, y_hat)) / (2 * y.size)

def main():
    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    y_hat1 = predict_(x1, theta1)
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

    print(type(y1)) # <class 'numpy.ndarray'>
    print(type(y_hat1)) # <class 'numpy.ndarray'>

    print(loss_elem_(y1, y_hat1)) # array([[0.], [1], [4], [9], [16]])
    print(loss_(y1, y_hat1)) # 3.0

    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)

    print(loss_(y2, y_hat2)) # 2.142857142857143
    print(loss_(y2, y2)) # 0.0

if __name__ == "__main__":
    main()