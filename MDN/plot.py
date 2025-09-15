import matplotlib.pyplot as plt
from utils import dscatter, regline

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
