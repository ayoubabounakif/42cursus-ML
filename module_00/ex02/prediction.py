import numpy as np

def simple_predict(x, theta):
    if len(x) == 0 or len(theta) == 0: return None
    if len(x.shape) != 1 or len(theta.shape) != 1: return None
    y_hat = np.array(theta[0] + theta[1] * x, dtype=float)
    return y_hat

def main():
    x = np.arange(1,6)
    theta1 = np.array([5, 0])
    print(simple_predict(x, theta1)) # array([5., 5., 5., 5., 5.])

    theta2 = np.array([0, 1])
    print(simple_predict(x, theta2)) # array([1., 2., 3., 4., 5.])

    theta3 = np.array([5, 3])
    print(simple_predict(x, theta3)) # array([ 8., 11., 14., 17., 20.])

    theta4 = np.array([-3, 1])
    print(simple_predict(x, theta4)) # array([-2., -1.,  0.,  1.,  2.])

if __name__ == '__main__':
    main()
    