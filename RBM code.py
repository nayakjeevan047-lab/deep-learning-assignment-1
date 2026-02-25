class RBM:
    def __init__(self, visible, hidden, lr=0.01):
        self.W = np.random.randn(visible, hidden) * 0.01
        self.hb = np.zeros(hidden)
        self.vb = np.zeros(visible)
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_h(self, v):
        prob = self.sigmoid(v @ self.W + self.hb)
        return prob, (prob > np.random.rand(*prob.shape)).astype(float)

    def sample_v(self, h):
        prob = self.sigmoid(h @ self.W.T + self.vb)
        return prob, (prob > np.random.rand(*prob.shape)).astype(float)

    def train(self, X, epochs=5):
        for epoch in range(epochs):
            for v0 in X:
                v0 = v0.reshape(1, -1)
                
                ph0, h0 = self.sample_h(v0)
                pv1, v1 = self.sample_v(h0)
                ph1, h1 = self.sample_h(v1)
                
                self.W += self.lr * (v0.T @ ph0 - v1.T @ ph1)
                self.vb += self.lr * (v0 - v1).flatten()
                self.hb += self.lr * (ph0 - ph1).flatten()
            
            print(f"Epoch {epoch+1} done")
