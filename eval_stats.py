import numpy as np

EPSILON = 1e-7


def r_squared(actual: np.ndarray, predicted: np.ndarray):
    rss = (_error(actual, predicted) ** 2).sum(1)
    tss = (_error(actual, actual.mean(1, keepdims=True)) ** 2).sum(1)
    r2 = 1.0 - rss / (tss + EPSILON)
    return r2.mean()


def _error(actual: np.ndarray, predicted: np.ndarray):
    """Simple error"""
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error

    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def mse(actual: np.ndarray, predicted: np.ndarray):
    """Mean Squared Error"""
    return np.mean(np.square(_error(actual, predicted)))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """Mean Absolute Error"""
    return np.mean(np.abs(_error(actual, predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """Symmetric Mean Absolute Percentage Error"""
    return np.mean(
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )
