import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib.machinery import SourceFileLoader
from sklearn.metrics import mean_squared_error

MyLinearRegression = SourceFileLoader("MyLinearRegression", "../ex03/my_linear_regression.py").load_module().MyLinearRegression

def predict_(x, theta):
  return theta[0] + theta[1] * x

def loss_elem_(y, y_hat):
    if (isinstance(y, np.ndarray) == False or isinstance(y_hat, np.ndarray) == False): return None
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape: return None
    return (y_hat - y) ** 2

def loss_(y, y_hat):
  if (isinstance(y, np.ndarray) == False or isinstance(y_hat, np.ndarray) == False): return None
  if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape: return None
  return np.sum(loss_elem_(y, y_hat)) / (2 * y.size)

def main():
  data = pd.read_csv("are_blue_pills_magics.csv")
  Xpill = np.array(data['Micrograms']).reshape(-1,1)
  Yscore = np.array(data['Score']).reshape(-1,1)

  theta = np.array([[89.0], [-8]])
  linear_model1 = MyLinearRegression(theta)
  linear_model2 = MyLinearRegression(np.array([[89.0], [-6]]))
  Y_model1 = linear_model1.predict_(Xpill)
  Y_model2 = linear_model2.predict_(Xpill)

  print(MyLinearRegression.mse_(Yscore, Y_model1)) # 57.603042857142825
  print(MyLinearRegression.mse_(Yscore, Y_model1) == mean_squared_error(Yscore, Y_model1)) # True
  print(MyLinearRegression.mse_(Yscore, Y_model2)) # 232.16344285714283
  print(MyLinearRegression.mse_(Yscore, Y_model2) == mean_squared_error(Yscore, Y_model2)) # True

  plt.plot(Xpill, Yscore, 'o')
  plt.plot(Xpill, Y_model1, '--g')
  plt.plot(Xpill, Y_model2, '--r')
  plt.show()


  print('---------- Plotting the loss function J(θ) in function of the θ values ----------')
  theta0 = np.linspace(80, 96, 6)
  theta1 = np.linspace(-14, -4, 100)
  for t0 in theta0:
    loss_array = []
    for t1 in theta1:
      y_hat = predict_(Xpill, np.array([[t0], [t1]]))
      loss_array.append(loss_(Yscore, y_hat))
    plt.plot(theta1, np.array(loss_array),
              label=f'J(\u03b8\u2080 = {t0:.2f},\u03b8\u2081)')
  plt.grid()
  plt.legend(loc='lower right')
  plt.xlabel('\u03b8\u2080')
  plt.ylabel(f"cost function J(\u03b8\u2080,\u03b8\u2081)")
  plt.ylim((10, 140))
  plt.show()


if __name__ == "__main__":
  main()