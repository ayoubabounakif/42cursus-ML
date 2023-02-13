import numpy as np

def simple_gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray): return None
    if x.size == 0 or y.size == 0 or theta.size == 0: return None
    if x.shape[0] != y.shape[0] or x.shape[1] != 1 or y.shape[1] != 1 or theta.shape[0] != 2 or theta.shape[1] != 1: return None
    nabla = np.zeros((2, 1))
    tmp_nabla = np.zeros((2, 1))
    m = x.shape[0]
    for i in range(m):
        hypothesis = theta[0] + theta[1] * x[i]
        tmp_nabla[0] = (hypothesis - y[i])
        tmp_nabla[1] = (hypothesis - y[i]) * x[i]
        nabla += tmp_nabla
    return nabla * (1 / m)

def main():
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(simple_gradient(x, y, theta1)) # array([[-19.0342574], [-586.66875564]])

    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(simple_gradient(x, y, theta2)) # array([[-57.86823748], [-2230.12297889]])

if __name__ == "__main__":
    main()