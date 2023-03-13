import numpy as np

class MyLinearRegression:
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    # Private Methods
    def __add_intercept(self, x, dtype=np.float64):
        if not isinstance(x, np.ndarray): return None
        if not (len(x) != 0): return None
        if len(x.shape) == 1:
            x = x.reshape((x.shape[0], 1))
        intercept = np.ones((x.shape[0], 1), dtype=dtype)
        return np.concatenate((intercept, x), axis=1)
    
    def __gradient_descent(self, x, y, theta):
        m = x.shape[0]
        X_ = self.__add_intercept(x)
        new_theta = theta
        for _ in range(self.max_iter):
            new_theta -= self.alpha * (1 / m) * X_.T.dot(X_.dot(new_theta) - y)
        return new_theta
    
    # Public Methods
    def fit_(self, x, y):
        if not isinstance(x, np.ndarray): return None
        if not isinstance(y, np.ndarray): return None
        if not (len(x) != 0): return None
        if not (len(y) != 0): return None
        if not (len(x) == len(y)): return None
        if not (len(x.shape) == 2): return None
        if not (len(y.shape) == 2): return None
        if not (y.shape[1] == 1): return None
        if not (x.shape[0] == y.shape[0]): return None
        self.thetas = self.__gradient_descent(x, y, self.thetas)
        return self.thetas
    
    def predict_(self, x):
        if not isinstance(x, np.ndarray): return None
        if not (len(x) != 0): return None
        if not (len(x.shape) == 2): return None
        x = self.__add_intercept(x)
        return np.dot(x, self.thetas)
    
    def loss_elem_(self, y, y_hat):
        if (isinstance(y, np.ndarray) == False or isinstance(y_hat, np.ndarray) == False): return None
        if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape: return None
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        if (isinstance(y, np.ndarray) == False or isinstance(y_hat, np.ndarray) == False): return None
        if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape: return None
        return np.sum(self.loss_elem_(y, y_hat)) / (2 * y.size)
    
    @staticmethod
    def mse_(y, y_hat):
        if (isinstance(y, np.ndarray) == False or isinstance(y_hat, np.ndarray) == False): return None
        if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape: return None
        return np.sum((y_hat - y) ** 2) / (y.size)

    
def main():
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])

    y_hat = mylr.predict_(X)
    print(y_hat) # array([[8.], [48.], [323.]])
    print(mylr.loss_elem_(Y, y_hat)) # array([[225.], [0.], [11025.]])
    print(mylr.loss_(Y, y_hat)) # 1875.0

    print('--' * 10)

    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    thetas = mylr.thetas
    np.set_printoptions(suppress=True)
    print(thetas) # array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])

    print('--' * 10)

    y_hat = mylr.predict_(X)
    print(y_hat) # array([[23.417..], [47.489..], [218.065...]])
    print(mylr.loss_elem_(Y, y_hat)) # array([[0.174..], [0.260..], [0.004..]])
    print(mylr.loss_(Y, y_hat)) # 0.0732..

if  __name__ == "__main__":
    main()
