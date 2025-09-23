import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from utils import dscatter, regline, density_kernel

def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(False)
    plt.show()
       
def plot_scatter(x, y, intercept=False, plot_regline=True, title='', xlabel='', ylabel='', 
                     xlim=None, ylim=None, s=10, return_params=False):
    if xlim is None:
        xlim = (x.min(), x.max())
    if ylim is None:
        ylim = (y.min(), y.max())

    # Compute density and regression line
    pre, post, dist = dscatter(x, y)

    plt.scatter(pre, post, c=dist, s=s)
    if plot_regline:
        xi, yi, slope, reg_intercept = regline(x, y, intercept=intercept)
        plt.plot(xi, yi, 'r--')

    # Format plot
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title)

    if return_params:
        return slope, reg_intercept

def plot_density(arr, xlim=None, ylim=None, xlabel='', ylabel='', title=''):
    if xlim is None:
        xlim = (arr[:,0].min(), arr[:,0].max())
    if ylim is None:
        ylim = (arr[:,1].min(), arr[:,1].max())
    
    dist, xmin, xmax, ymin, ymax = density_kernel(arr)
    plt.imshow(np.rot90(dist), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    
    # Format plot
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
def plot_pdf(y, xlim=None, ylim=None, xlabel='', ylabel='', title='', label='', legend=False):   
    xi = np.linspace(y.min(), y.max(), 100)
    dist = gaussian_kde(y)(xi)
    
    plt.plot(xi, dist, label=label)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Format plot
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if legend:
        plt.legend()