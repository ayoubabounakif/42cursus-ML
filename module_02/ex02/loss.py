import numpy as np

def loss_(y, y_hat):
    m = y.shape[0]
    return (1 / (2 * m) * np.dot((y_hat - y).T, (y_hat - y)).item())

def main():
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(loss_(X, Y)) # 2.142857142857143
    print(loss_(X, X)) # 0.0

if __name__ == '__main__':
    main()