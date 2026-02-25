import numpy as np
import matplotlib.pyplot as plt


# LOAD MNIST CSV DATA

def load_mnist_csv(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    X = data[:, 1:] / 255.0
    y = data[:, 0].astype(int)
    return X, y

X_train, y_train = load_mnist_csv("mnist_train.csv")
X_test, y_test = load_mnist_csv("mnist_test.csv")

# Use subset (faster)
X_train = X_train[:10000]
y_train = y_train[:10000]


# UTIL FUNCTIONS

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)

def sigmoid(x): return 1 / (1 + np.exp(-x))

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

# MLP MODEL

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=10, batch_size=64):
        y = one_hot(y)
        losses = []

        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                Xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]

                preds = self.forward(Xb)
                loss = cross_entropy(preds, yb)
                self.backward(Xb, yb)

            losses.append(loss)
            print(f"MLP Epoch {epoch+1}, Loss: {loss:.4f}")

        return losses

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


# AUTOENCODER

class Autoencoder:
    def __init__(self, input_size, hidden_size, lr=0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, input_size) * 0.01
        self.b2 = np.zeros((1, input_size))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def loss(self, X, out, lam=0.001):
        recon = np.mean((X - out) ** 2)
        sparse = lam * np.mean(np.abs(self.A1))
        return recon + sparse

    def backward(self, X, out):
        m = X.shape[0]

        dZ2 = (out - X) * out * (1 - out)
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.A1 * (1 - self.A1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, epochs=10):
        losses = []
        for e in range(epochs):
            out = self.forward(X)
            loss = self.loss(X, out)
            self.backward(X, out)
            losses.append(loss)
            print(f"AE Epoch {e+1}, Loss: {loss:.4f}")
        return losses


# RBM

class RBM:
    def __init__(self, visible, hidden, lr=0.01):
        self.W = np.random.randn(visible, hidden) * 0.01
        self.hb = np.zeros(hidden)
        self.vb = np.zeros(visible)
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_h(self, v):
        p = self.sigmoid(v @ self.W + self.hb)
        return p, (p > np.random.rand(*p.shape)).astype(float)

    def sample_v(self, h):
        p = self.sigmoid(h @ self.W.T + self.vb)
        return p, (p > np.random.rand(*p.shape)).astype(float)

    def train(self, X, epochs=5):
        for e in range(epochs):
            for v0 in X[:1000]:
                v0 = v0.reshape(1, -1)
                ph0, h0 = self.sample_h(v0)
                pv1, v1 = self.sample_v(h0)
                ph1, _ = self.sample_h(v1)

                self.W += self.lr * (v0.T @ ph0 - v1.T @ ph1)

            print(f"RBM Epoch {e+1} done")





# MLP
mlp = MLP(784, 128, 10, lr=0.01)
mlp_loss = mlp.train(X_train, y_train, epochs=10)

preds = mlp.predict(X_test[:2000])
acc = np.mean(preds == y_test[:2000])
print("MLP Accuracy:", acc)

# Autoencoder
ae = Autoencoder(784, 64)
ae_loss = ae.train(X_train[:5000], epochs=10)

# RBM
rbm = RBM(784, 64)
rbm.train(X_train[:1000], epochs=5)


# PLOTS

plt.plot(mlp_loss)
plt.title("MLP Loss")
plt.show()

plt.plot(ae_loss)
plt.title("Autoencoder Loss")
plt.show()
