
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, denoised_X, X):
    predicted = model.predict(denoised_X)
    mse = mean_squared_error(X.flatten(), predicted.flatten())
    r2 = r2_score(X.flatten(), predicted.flatten())
    print(f"🔍 Evaluation: MSE = {mse:.4f}, R2 = {r2:.4f}")
