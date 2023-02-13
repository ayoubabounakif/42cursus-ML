import numpy as np

class MyLinearRegression:
  def __init__(self, thetas, alpha=0.001, max_iter=1000):
    self.alpha = alpha
    self.max_iter = max_iter
    self.thetas = thetas

  # Private Methods
  def __add_intercept(self, x, dtype=np.float64):
    if not isinstance(x, np.ndarray): return None
    if not (len(x) != 0): return None
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    intercept = np.ones((x.shape[0], 1), dtype=dtype)
    return np.concatenate((intercept, x), axis=1)

  def __simple_gradient(self, x, y, theta):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray): return None
    if x.size == 0 or y.size == 0 or theta.size == 0: return None
    if x.shape[0] != y.shape[0] or x.shape[1] != 1 or y.shape[1] != 1 or theta.shape[0] != 2 or theta.shape[1] != 1: return None
    # ∇(J) = 1/m * X'T * (X'θ - y)
    m = x.shape[0]
    X_ = self.__add_intercept(x)
    nabla = (1 / m) * np.dot(X_.T, np.dot(X_, theta) - y)
    return nabla

  
  # Public Methods
  def fit_(self, x, y):
    if (isinstance(x, np.ndarray) == False or isinstance(y, np.ndarray) == False or isinstance(self.thetas, np.ndarray) == False): return None
    if (x.size == 0 or y.size == 0 or self.thetas.size == 0 or x.shape[1] != 1 or y.shape[1] != 1 or self.thetas.shape[0] != 2): return None
    if (isinstance(self.alpha, float) == False or isinstance(self.max_iter, int) == False): return None
    if (self.max_iter <= 0 or self.alpha <= 0): return None

    for _ in range(self.max_iter):
      # print(theta)
      self.thetas -= self.alpha * self.__simple_gradient(x, y, self.thetas)
    return self.thetas

  def predict_(self, x):
    if len(x) == 0 or len(self.thetas) == 0: return None
    x_ = self.__add_intercept(x)
    y_hat = np.dot(x_, self.thetas)
    return y_hat

  def loss_elem_(self, y, y_hat):
    if (isinstance(y, np.ndarray) == False or isinstance(y_hat, np.ndarray) == False): return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape: return None
    return (y_hat - y) ** 2

  def loss_(self, y, y_hat):
    if (isinstance(y, np.ndarray) == False or isinstance(y_hat, np.ndarray) == False): return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape: return None
    return np.sum(self.loss_elem_(y, y_hat)) / (2 * y.size)

  # Static Methods
  @staticmethod
  def mse_(y, y_hat):
    if (isinstance(y, np.ndarray) == False or isinstance(y_hat, np.ndarray) == False): return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape: return None
    return np.sum((y_hat - y) ** 2) / (y.size)
    
def main():
  x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
  y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

  print('------- Example 00 --------')
  lr1 = MyLinearRegression(np.array([[2], [0.7]]))
  y_hat = lr1.predict_(x) # array([[10.74695094], [17.05055804], [24.08691674], [36.24020866], [42.25621131]])
  print(y_hat)
  print(lr1.loss_elem_(y, y_hat)) # array([[710.45867381], [364.68645485], [469.96221651],[108.97553412], [299.37111101]])
  print(lr1.loss_(y, y_hat)) # 195.34539903032385

  print('------- Example 01 --------')
  lr2 = MyLinearRegression(np.array([[1], [1]], dtype=np.float64), 5e-8, 1500000)
  lr2.fit_(x, y)
  print(lr2.thetas) # array([[1.40709365], [1.1150909 ]])
  y_hat = lr2.predict_(x)
  print(y_hat) # array([[15.3408728 ], [25.38243697], [36.59126492], [55.95130097], [65.53471499]])
  print(lr2.loss_elem_(y, y_hat)) # array([[486.66604863], [115.88278416], [ 84.16711596], [ 85.96919719], [ 35.71448348]])
  print(lr2.loss_(y, y_hat)) # 80.83996294128525


if __name__ == "__main__":
  main()