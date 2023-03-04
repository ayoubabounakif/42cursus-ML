import numpy as np
from importlib.machinery import SourceFileLoader

simple_gradient = SourceFileLoader("vec_gradient", "../ex01/vec_gradient.py").load_module().simple_gradient

def add_intercept(x, dtype=np.float64):
    if not isinstance(x, np.ndarray): return None
    if not (len(x) != 0): return None
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    intercept = np.ones((x.shape[0], 1), dtype=dtype)
    return np.concatenate((intercept, x), axis=1)

def predict_(x, theta):
    if len(x) == 0 or len(theta) == 0: return None
    # if len(x.shape) != 1 or len(theta.shape) != 2: return None
    x_ = add_intercept(x)
    y_hat = np.dot(x_, theta)
    return y_hat

# def predict_(x, theta, dtype=np.float64):
#     y_hat = np.array(theta[0] + theta[1] * x, dtype=dtype)
#     return y_hat

def fit_(x, y, theta, alpha, max_iter):
  if (isinstance(x, np.ndarray) == False or isinstance(y, np.ndarray) == False or isinstance(theta, np.ndarray) == False): return None
  if (x.size == 0 or y.size == 0 or theta.size == 0 or x.shape[1] != 1 or y.shape[1] != 1 or theta.shape[0] != 2): return None
  if (isinstance(alpha, float) == False or isinstance(max_iter, int) == False): return None
  if (max_iter <= 0 or alpha <= 0): return None

  for _ in range(max_iter):
    # print(theta)
    theta -= alpha * simple_gradient(x, y, theta)
  return theta


def main():
  x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
  y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

  theta = np.array([1, 1], dtype=np.float64).reshape((-1, 1))

  theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)

  print('----- Calculations Done ------')
  print(f'theta1:\n{theta1}') # array([[1.40709365], [1.1150909 ]])
  print('-----')
  print(f'predict_(x, theta1):\n{predict_(x, theta1)}') # array([[15.3408728 ], [25.38243697], [36.59126492], [55.95130097], [65.53471499]])


if __name__ == "__main__":
  main()