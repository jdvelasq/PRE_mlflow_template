from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(model, x, y):
    """Calculate and return MSE, MAE, and R2 metrics."""

    y_pred = model.predict(x)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, mae, r2
