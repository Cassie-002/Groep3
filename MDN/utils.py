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
