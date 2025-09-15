import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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

def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(False)
    plt.show()
    
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