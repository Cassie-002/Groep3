import numpy as np
import json
from scipy.stats import gaussian_kde

def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
        
def open_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def dscatter(x,y):
    xy = np.vstack([x,y])
    c = gaussian_kde(xy)(xy)
    idx = c.argsort()
    x, y, c = x[idx], y[idx], c[idx]
    return x, y, c

def inv_sigmoid(x):
    return np.log((x)/(1-(x)))

def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig
    
def regline(x, y, intercept=True):
    if not intercept:
        slope = np.sum(x * y) / np.sum(x**2)
        y_hat = slope * x
        return x, y_hat, slope, 0
        
    x_mean, y_mean = np.mean(x), np.mean(y)
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    c = y_mean - slope * x_mean
    y_hat = c + slope * x
    return x, y_hat, slope, c

def relative_error(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true)

def density_kernel(data):
    """Performs a Gaussian kernel density estimation on 2D data.

    Args:
        data (Array-like)): First column is x values, second column is y values.

    Returns:
        Z (2D array): Density values on a grid.
        xmin, xmax, ymin, ymax (float): Extents of the grid.
    """
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    kernel = gaussian_kde(data.T)
    Z = np.reshape(kernel(positions).T, X.shape)

    return Z, xmin, xmax, ymin, ymax

def combine_pre_post(x,y):
    return np.vstack((x,y)).T

