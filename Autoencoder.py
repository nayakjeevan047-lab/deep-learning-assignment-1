class Autoencoder:
    def __init__(self, input_size, hidden_size, lr=0.01):
        self.lr = lr
        
        
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        
        
        self.W2 = np.random.randn(hidden_size, input_size) * 0.01
        self.b2 = np.zeros((1, input_size))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = sigmoid(self.Z1)  # latent
        
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)  # reconstruction
        
        return self.A2

    def loss(self, X, output, lam=0.0):
        recon_loss = np.mean((X - output) ** 2)
        sparse_penalty = lam * np.mean(np.abs(self.A1))
        return recon_loss + sparse_penalty

    def backward(self, X, output):
        m = X.shape[0]
        
        dZ2 = (output - X) * output * (1 - output)
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

    def train(self, X, epochs=10, lam=0.0):
        losses = []
        
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.loss(X, output, lam)
            self.backward(X, output)
            
            losses.append(loss)
            print(f"Epoch {epoch+1}, Loss: {loss}")
        
        return losses
