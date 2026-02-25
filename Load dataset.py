import numpy as np

def load_mnist_csv(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    
    X = data[:, 1:] / 255.0  
    y = data[:, 0].astype(int)
    
    return X, y


X_train, y_train = load_mnist_csv("mnist_train.csv")
X_test, y_test = load_mnist_csv("mnist_test.csv")
