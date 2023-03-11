import numpy as np

def simple_predict(x, theta):
    m = x.shape[0]
    y_hat = np.zeros((m, 1))
    for i in range(m):
        y_hat[i] = np.sum(theta.T[:,1:] * x[i]) + theta[0]
    return y_hat
        
def main():
    x = np.arange(1,13).reshape((4,-1))
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
    print(simple_predict(x, theta1)) # array([[5.], [5.], [5.], [5.]])

    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    print(simple_predict(x, theta2)) # array([[ 1.], [ 4.], [ 7.], [10.]])

    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    print(simple_predict(x, theta3)) # array([[ 9.64], [24.28], [38.92], [53.56]])

    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    print(simple_predict(x, theta4)) # array([[12.5], [32. ], [51.5], [71. ]])

if __name__ == '__main__':
    main()