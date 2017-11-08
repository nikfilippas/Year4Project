"""
Useful function definitions.
"""

import numpy as np
from scipy.optimize import leastsq
from astropy.io import fits
from astropy.wcs import WCS


class fluxcount:
    """Measures fluxes in a given data array.
    """

    def __init__(self, data):
        self.data = data
        self.size = len(data)  # side length of square cut-out


    def dist(self, data):
        """ Calculates the euclidean distance between a pixel and the center
        of an image.
        """
        # locates central coordinates
        cents = np.argwhere(self.data == np.max(self.data))
        cpix = np.array([np.round(np.mean(cents[:, i])) for i in range(2)])
        self.Xc, self.Yc = cpix.astype(int)  # central pixel coordinates

        distarray = np.zeros((self.size, self.size))  # square 2d array
        for i in range(self.size):
            for j in range(self.size):
                w = np.array([i,j])  # position of pixel
                distarray[i,j] = np.linalg.norm(w - cpix)  # Euclidean distance

        return distarray


    def light_radius(self, p=0.5):
        """ Calculates the radius which contains ``p`` of the total flux.
        """
        total_light = self.data.sum()  # total integrated flux
        distarray = self.dist(self.data)  # distance array

        self.max_dist = np.min([
                self.Xc, self.size-self.Xc,
                self.Yc, self.size-self.Yc
                ])  # nearest edge
        distances = np.sqrt(np.arange(1, (self.max_dist+1)**2, 1))  # distances

        intflux = 0  # integrated flux
        for d in distances:
            intflux = self.data[distarray <= d].sum()  # integrated flux in d
            if intflux >= p*total_light:
                break

        return d


    def FWHM(self):
        """ Calculates the full width at half maximum of a 2D array.
        """
        HM = self.data.max()/2  # half max

        distances = np.arange(1, self.max_dist+1, 1)  # distances
        for d in distances:
            px = np.round(d)
            val = self.data[self.Xc+px, self.Yc+px]
            if val <= HM:
                break

        return 2*d


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
