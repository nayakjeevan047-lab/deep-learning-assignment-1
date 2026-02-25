def detect_outliers(model, X, threshold):
    recon = model.forward(X)
    error = np.mean((X - recon)**2, axis=1)
    return error > threshold
