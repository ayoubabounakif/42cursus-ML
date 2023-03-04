import numpy as np
from importlib.machinery import SourceFileLoader

add_intercept = SourceFileLoader("tools", "../../module_00/ex03/tools.py").load_module().add_intercept

def simple_gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray): return None
    if x.size == 0 or y.size == 0 or theta.size == 0: return None
    if x.shape[0] != y.shape[0] or x.shape[1] != 1 or y.shape[1] != 1 or theta.shape[0] != 2 or theta.shape[1] != 1: return None
    # ∇(J) = 1/m * X'T * (X'θ - y)
    m = x.shape[0]
    X_ = add_intercept(x)
    nabla = (1 / m) * np.dot(X_.T, np.dot(X_, theta) - y)
    return nabla

def main():
    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))

    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(simple_gradient(x, y, theta1)) # array([[-19.0342574], [-586.66875564]])

    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(simple_gradient(x, y, theta2)) # array([[-57.86823748], [-2230.12297889]])

if __name__ == "__main__":
    main()