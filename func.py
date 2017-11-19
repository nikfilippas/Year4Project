"""
Useful function definitions.
"""

import numpy as np
from scipy.optimize import leastsq


def gaussian(A, x0, y0, s):
    """Returns a symmetric gaussian function with the given parameters.
    """
    g = lambda x, y: A*np.exp(-( ((x0-x)/s)**2 + ((y0-y)/s)**2) /2)
    return g


def moments(data):
    """Returns (A, x0, y0, s) the gaussian parameters of a symmetric 2D
    distribution by calculating its moments.
    """
    total = data.sum()
    X, Y = np.indices(data.shape)  # creates numbered mesh

    x = (X*data).sum()/total
    y = (Y*data).sum()/total

    col = data[:, int(y)]
    row = data[int(x), :]

    sx = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    sy = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())

    s = np.sqrt(sx**2 + sy**2)  # quadrature std

    A = data.max()
    return A, x, y, s


def fitgaussian(data):
    """Returns (A, x0, y0, s) the gaussian parameters of a symmetric 2D
    distribution found by a fit.
    """
    params = moments(data)
    errorf = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = leastsq(errorf, params)
    return p
