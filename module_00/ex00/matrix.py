import numpy as np

class Matrix:

  def __init__(self, data):
    if isinstance(data, list) and all(isinstance(i, list) for i in data):
      self.data = data
      self.shape = (len(data), len(data[0]))
      return
    elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], int) and isinstance(data[1], int):
      self.data = [[0] * data[1] for i in range(data[0])]
      self.shape = data
      return
    else:
      raise TypeError("Invalid data type")
  
  def __str__(self) -> str:
    return str(self.data)

  def __repr__(self) -> str:
    return str(self.data)

  def __add__(self, other):
    if isinstance(other, Matrix) and len(self.data) == len(other.data) and len(self.data[0]) == len(other.data[0]):
      return Matrix([[self.data[i][j] + other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))])
    else:
      raise TypeError("Invalid addition for matrix")

  def __radd__(self, other):
    self.__add__(other)

  def __sub__(self, other):
    if isinstance(other, Matrix) and len(self.data) == len(other.data) and len(self.data[0]) == len(other.data[0]):
      return Matrix([[self.data[i][j] - other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))])
    else:
      raise TypeError("Invalid data type")

  def __rsub__(self, other):
    self.__sub__(other)

  def __mul__(self, other):
    if isinstance(other, Matrix) and len(self.data[0]) == len(other.data):
      return Matrix([[sum([self.data[i][k] * other.data[k][j] for k in range(len(self.data[0]))]) for j in range(len(other.data[0]))] for i in range(len(self.data))])
    else:
      raise TypeError("Invalid data type")
    
  def __rmul__(self, other):
    self.__mul__(other)

  def __truediv__(self, other):
    if isinstance(other, Matrix) and len(self.data) == len(other.data) and len(self.data[0]) == len(other.data[0]):
      return Matrix([[self.data[i][j] / other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))])
    else:
      raise TypeError("Invalid data type")

  def __rtuediv__(self, other):
    self.__truediv__(other)

  def T(self):
    return Matrix([[self.data[j][i] for j in range(len(self.data))] for i in range(len(self.data[0]))])

class Vector(Matrix):
  
  def __init__(self, data) -> None:
    if isinstance(data, list) and all(isinstance(i, (float, int)) for i in data):
      super().__init__([data])
      return
    elif isinstance(data, list) and all(isinstance(i, list) for i in data) and all(len(i) == 1 for i in data):
      super().__init__(data)
      return
    elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], int) and isinstance(data[1], int):
      super().__init__(data)
      return
    else:
      raise TypeError("Invalid data type")

  def dot(self, other):
    if isinstance(other, Vector) and len(self.data[0]) == len(other.data[0]):
      return sum([self.data[i][0] * other.data[i][0] for i in range(len(self.data))])
    else:
      raise TypeError("Invalid data type")

def main():

  v1 = Vector([1, 2, 3]) # ccreate a row vector
  v2 = Vector([[1], [2], [3]]) # create a column vector
  # v3 = Vector([[1, 2], [3, 4]]) # error
  # print(v3)

  print(v1.shape, v2.shape)

  # print(m1 + m2)

  m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
  print(f'm1 shape --> {m1.shape}') # (3, 2)
  print(f'm1 transpose --> {m1.T()}') # Matrix([[0., 2., 4.], [1., 3., 5.]])
  print(f'm1 transpose shape --> {m1.T().shape}') # (2, 3)

  m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
  print(f'm1 shape --> {m1.shape}') # (2, 3)
  print(f'm1 transpose --> {m1.T()}') # Matrix([[0., 1.], [2., 3.], [4., 5.]])
  print(f'm1 transpose shape --> {m1.T().shape}') # (3, 2)

  m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
              [0.0, 2.0, 4.0, 6.0]])

  m2 = Matrix([[0.0, 1.0],
              [2.0, 3.0],
              [4.0, 5.0],
              [6.0, 7.0]])
  print(f'multiplication --> {m1 * m2}') # Matrix([[28.0, 34.0], [56.0, 70.0]])

  m1 = Matrix([[0.0, 1.0, 2.0],
              [0.0, 2.0, 4.0]])
  v1 = Vector([[1], [2], [3]])
  print(f'multiplication --> {m1 * v1}') # Matrix([[8.0], [16.0]])
  print(f'type --> {type(m1 * v1)}') # <class '__main__.Matrix'>

  v1 = Vector([[1], [2], [3]])
  v2 = Vector([[2], [4], [8]])
  print(f'v1 + v2 --> {v1 + v2}') # Matrix([[3], [6], [11]])

  print(f'v1 dot v2 --> {v1.dot(v2)}') # 34.0

  # print(v1.dot(v3)) # error


if __name__ == '__main__':
  main()


    
