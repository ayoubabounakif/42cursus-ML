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
