import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def dscatter(x,y):
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x,y,c=z, s=10)
    
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
