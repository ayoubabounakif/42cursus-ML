import numpy as np
from importlib.machinery import SourceFileLoader
import matplotlib.pyplot as plt

loss_ = SourceFileLoader("loss_", "../ex07/vec_loss.py").load_module().loss_

def plot_with_loss(x, y, theta):
  y_hat = theta[0] + theta[1] * x
  plt.title("Loss: " + str(loss_(y, y_hat)))
  plt.plot(x, y, 'o')
  plt.plot(x, y_hat)
  for i in range(len(x)):
    y_hat = theta[0] + theta[1] * x[i]
    plt.plot([x[i], x[i]], [y[i], y_hat], ':')
  plt.show()

def main():

  x = np.arange(1,6)
  y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

  theta1= np.array([18,-1])
  plot_with_loss(x, y, theta1)

  theta2 = np.array([14, 0])
  plot_with_loss(x, y, theta2)

  theta3 = np.array([12, 0.8])
  plot_with_loss(x, y, theta3)

if __name__ == "__main__":
  main()