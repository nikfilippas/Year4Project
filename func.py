"""
Useful function definitions.
"""

import numpy as np
from scipy.optimize import leastsq
from matplotlib.path import Path


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


def arraydist(verts):
    """Computes the euclidean distance between consecutive data points.
    Also includes the distance between the final and the first data point.
    """
    verts_x, verts_y = verts.T  # (x,y) components
    dist = np.array([np.sqrt((verts_x[i+1] - verts_x[i])**2 +
                             (verts_y[i+1] - verts_y[i])**2)
                            for i in range(len(verts)-1)])
    dist_init = np.sqrt((verts_x[0] - verts_x[-1])**2 +
                        (verts_y[0] - verts_y[-1])**2)  # last from first
    dist = np.append(dist, dist_init)

    return dist


def polymask(img, verts):
    """Masks the points within the given polygon vertices and returns the
    masked image. Vertices should be given in (x,y) format.
    """
    ny, nx = img.shape

    # Creates vertex coordinates for each grid cell
    x, y = np.meshgrid(np.arange(1, nx+1), np.arange(1, ny+1))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    path = Path(verts)  # path object (polygon)
    grid = path.contains_points(points)
    grid = grid.reshape((nx, ny))
    grid = grid.astype(int)  # integer booleans

    return grid


def imgmask(img, verts, dist_cutoff=10):
    """Masks images with multiple masking regions. Masks each region separately
     and combines masks. `dist_cutoff` is the region-separating pixel distance.
     """
    dist = arraydist(verts)  # consecutive distances

    if len(dist >= dist_cutoff) != 0:
        cut = np.where(dist >= dist_cutoff)[0]  # region bounds
        regions = np.split(verts, cut)

        # supress first vertex
        for i in range(1, len(regions)):
            regions[i] = regions[i][1:]
        # merge last with first
        regions[0] = np.concatenate((regions[-1], regions[0]))
        del regions[-1]

        # mask each region separately
        ny, nx = img.shape  # dimensions
        nz = len(regions)  # dimensions
        grid = np.zeros((nz, ny, nx))
        for i, reg in enumerate(regions):
            grid[i] = polymask(img, reg)

        # sum masks
        grid = np.sum(grid, axis=0)

    else:
        grid = polymask(img, verts)

    return grid
