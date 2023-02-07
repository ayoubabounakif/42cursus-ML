import numpy as np

# 1/2m (ˆy − y) · (ˆy − y)
def loss_(y, y_hat):
  if (isinstance(y, np.ndarray) == False or isinstance(y_hat, np.ndarray) == False): return None
  if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape: return None
  squaurd_sum = np.sum((y_hat - y) ** 2)
  return (1 / (2 * y.size)) * squaurd_sum

def main():
  X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
  Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

  print(loss_(X, Y)) # 2.142857142857143

  print(loss_(X, X)) # 0.0

if __name__ == "__main__":
  main()
