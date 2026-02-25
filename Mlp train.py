mlp = MLP(784, 128, 10, lr=0.01)

mlp_loss = mlp.train(X_train, y_train, epochs=10)

preds = mlp.predict(X_test[:2000])
acc = np.mean(preds == y_test[:2000])

print("MLP Accuracy:", acc)
