import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def __check_input(y, y_hat):
  """
  Check if the input is valid.

  Parameters:
  y: has to be an numpy.ndarray, a vector of dimension m * 1.
  y_hat: has to be an numpy.ndarray, a vector of dimension m * 1.

  Returns:
  True if the input is valid, False otherwise.
  """
  if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray): return False
  if len(y) == 0 or len(y_hat) == 0: return False
  if y.shape != y_hat.shape: return False
  return True

def mse_(y, y_hat):
  if not __check_input(y, y_hat): return None
  return np.sum((y - y_hat) ** 2) / len(y)

def rmse_(y, y_hat):
  if not __check_input(y, y_hat): return None
  return np.sqrt(mse_(y, y_hat))

def mae_(y, y_hat):
  if not __check_input(y, y_hat): return None
  return np.sum(np.abs(y - y_hat)) / len(y)

def r2score_(y, y_hat):
  if not __check_input(y, y_hat): return None
  return 1 - (np.sum((y_hat - y) ** 2) / np.sum((y - np.mean(y)) ** 2))

def main():
  x = np.array([0, 15, -9, 7, 12, 3, -21])
  y = np.array([2, 14, -13, 5, 12, 4, -19])

  print(mse_(x, y)) # 4.285714285714286
  print(rmse_(x, y)) # 2.0701966780270626
  print(mae_(x, y)) # 1.7142857142857142
  print(r2score_(x, y)) # 0.9681721733858745

  print('----------------')

  print(mean_squared_error(x, y)) # 4.285714285714286
  print(np.sqrt(mean_squared_error(x, y))) # 2.0701966780270626
  print(mean_absolute_error(x, y)) # 1.7142857142857142
  print(r2_score(x, y)) # 0.9681721733858745

if __name__ == "__main__":
  main()