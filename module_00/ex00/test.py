from matrix import Matrix
from matrix import Vector

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
