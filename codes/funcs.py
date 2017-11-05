"""

"""

import numpy as np
from scipy.optimize import curve_fit


def gaussian(data, A, x0, y0, s):
    """Returns a symmetric gaussian function with the given parameters.
    """
    x, y = np.indices(data.shape)
    g = lambda x, y: A*np.exp(-( ((x0-x)/s)**2 + ((y0-y)/s)**2) /2)
    return g



def fitgaussian(data):
    """Returns (A, x0, y0, sx, sy) the gaussian parameters of a 2D distribution
    found by a fit.
    """
    A = data.max()  # amplitude
    x0, y0 = np.unravel_index(np.argmax(data), data.shape)  # maximum coords
    p_guess = [A, x0, y0, 1, 1]  # guess params

    x, y = np.indices(data.shape)  # pixel values
    xy = np.column_stack((x, y))

    popt, pcov = curve_fit(gaussian, xy, data, p0=p_guess)

    return popt
