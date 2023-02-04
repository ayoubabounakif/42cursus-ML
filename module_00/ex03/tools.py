import numpy as np

def add_intercept(x, dtype=np.float64):
    if not isinstance(x, np.ndarray): return None
    if not (len(x) != 0): return None
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    intercept = np.ones((x.shape[0], 1), dtype=dtype)
    return np.concatenate((intercept, x), axis=1)

if __name__ == "__main__":
    x = np.arange(1,6)
    print(x)
    print('----')
    print(add_intercept(x))

    print('----')

    y = np.arange(1,10).reshape((3,3))
    print(y)
    print('----')
    print(add_intercept(y))