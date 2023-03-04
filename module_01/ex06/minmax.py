import numpy as np

# formulae: x'[i] = (x[i] - min(x)) / (max(x) - min(x))     # for i = 1, 2, ..., m
def minmax(x):
  return (x - x.min()) / (x.max() - x.min())

def main():
  X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
  print(minmax(X)) # array([0.58333333, 1. , 0.33333333, 0.77777778, 0.91666667, 0.66666667, 0. ])

  print('--------------')

  Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
  print(minmax(Y)) # array([0.63636364, 1. , 0.18181818, 0.72727273, 0.93939394, 0.6969697 , 0. ])

if __name__ == "__main__":
  main()