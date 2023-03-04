import numpy as np

# x'[i] = (x[i] - 1 / m * sum(x[i]))
#         ----------------------------
#         sqrt(1 / m - 1 * sum(x[i] - 1 / m * sum(x[i])) ** 2)      # for i in 1, 2, ..., m

# x' = (x - np.mean(x)) / np.std(x)
def zscore(x):
  if not isinstance(x, np.ndarray) or x.size == 0: return None
  return (x - np.mean(x)) / np.std(x)

def main():
  X = np.array([0, 15, -9, 7, 12, 3, -21])
  print(zscore(X)) # array([-0.08620324, 1.2068453 , -0.86203236, 0.51721942, 0.94823559, 0.17240647, -1.89647119])

  Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
  print(zscore(Y)) # array([ 0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659, 0.28795027, -1.72770165])

if __name__ == "__main__":
  main()