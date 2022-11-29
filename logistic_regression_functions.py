import numpy as np


def predict_proba(X, coeffs):
    """Calculate the predicted conditional probabilities (floats between 0 and
    1) for the given data with the given coefficients.

    Parameters
    ----------
    X: np.array of shape (n_samples, n_features)  
        The data (independent variables) to use for prediction.
    coeffs: np.array of shape (n_features, )
        The hypothesized coefficients.  Note that the shape of X, and coeffs

    Returns
    -------
    predicted_probabilities: 
        The conditional probabilities given the data and coefficients.
    """
    return 1. / (1. + np.exp( -np.dot(X, coeffs) ))
    


def predict(X, coeffs, thresh=0.5):
    """
    Calculate the predicted class labels (0 or 1) for the given data with the
    given coefficients by comparing the predicted probabilities to a given
    threshold.

    Parameters
    ----------
    X: np.array of shape (n_samples, n_features)  
        The data (independent variables) to use for prediction.
    coeffs: np.array of shape (n_features, )
        The hypothesized coefficients.  Note that the shape of X, and coeffs
        must align.
    thresh: float
        Threshold for classification.

    Returns
    -------
    predicted_class: int
        The predicted class.
    """
    predicted_classes = np.where(predict_proba(X, coeffs) >= thresh, 1, 0)

    return predicted_classes


def cost(X, y, coeffs):
    """
    Calculate the total logistic cost function of the data with the given
    coefficients.

    Parameters
    ----------
    X: np.array of shape (n_samples, n_features)  
        The data (independent variables) to use for prediction.
    y: np.array of shape (n_samples, )
        The actual class values of the response.  Must be encoded as 0's and
        1's.  Also, must align properly with X and coeffs.
    coeffs: np.array of shape (n_features, )
        The hypothesized coefficients.  Note that the shape of X, y, and coeffs
        must align.

    Returns
    -------
    logistic_cost: float
        The value of the logistic cost function evaluated at the given
        coefficients.
    """
    m = X.shape[0]
    total_cost = -(1 / m) * np.sum(y * np.log(predict_proba(X,coeffs)) + (1 - y) * np.log(1 - predict_proba(X,coeffs)))
    return total_cost

def gradient(X, y, coeffs):
    """
    Calculate the gradient of the logistic cost function with the given
    coefficients.

    Parameters
    ----------
    X: np.array of shape (n_samples, n_features)  
        The data (independent variables) to use for prediction.
    y: np.array of shape (n_samples, )
        The actual class values of the response.  Must be encoded as 0's and
        1's.  Also, must align properly with X and coeffs.
    coeffs: np.array of shape (n_features, )
        The hypothesized coefficients.  Note that the shape of X, y, and coeffs
        must align.

    Returns
    -------
    logistic_gradient: np.array of shape (n_features, )
        The gradient of the logistic cost function evaluated at the given
        coefficients.
    """
    m = X.shape[0]
    return (1 / m) * np.dot(X.T,  predict_proba(X, coeffs) - y)
