import numpy as np

def predict_(x, theta):
    m = x.shape[0]
    x = np.c_[np.ones((m, 1)), x]
    return np.dot(x, theta)

def fit_(x, y, theta, alpha, max_iter):
    m = x.shape[0]
    X_ = np.c_[np.ones((m, 1)), x]
    new_theta = theta
    for _ in range(max_iter):
        new_theta -= alpha * (1 / m) * X_.T.dot(X_.dot(new_theta) - y)
    return new_theta

def main():
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])

    theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
    print(theta2) # array([[41.99..],[0.97..], [0.77..], [-1.20..]])

    print('--' * 10)

    print(predict_(x, theta2)) # array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])

if __name__ == '__main__':
    main()