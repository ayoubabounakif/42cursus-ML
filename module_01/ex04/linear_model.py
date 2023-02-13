import pandas as pd
import numpy as np
# from sklearn.metrics import mean_squared_error
from importlib.machinery import SourceFileLoader

MyLinearRegression = SourceFileLoader("MyLinearRegression", "../ex03/my_linear_regression.py").load_module().MyLinearRegression


def main():
  data = pd.read_csv("are_blue_pills_magics.csv")
  Xpill = np.array(data['Micrograms']).reshape(-1,1)
  Yscore = np.array(data['Score']).reshape(-1,1)

  linear_model1 = MyLinearRegression(np.array([[89.0], [-8]]))
  linear_model2 = MyLinearRegression(np.array([[89.0], [-6]]))
  Y_model1 = linear_model1.predict_(Xpill)
  Y_model2 = linear_model2.predict_(Xpill)

  print(MyLinearRegression.mse_(Yscore, Y_model1)) # 57.60304285714282
  print(MyLinearRegression.mse_(Yscore, Y_model2)) # 232.16344285714285



if __name__ == "__main__":
  main()