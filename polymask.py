"""
Masks part of galaxy image contained within some polygon vertices.
Polygon vertices are exported using the Aladin Sky Atlas 'draw' tool.
Handles multiple masking regions.
"""

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import func

drawing = input("Enter the path to the Aladin 'Drawing.txt' file:\n")
image = input("Enter the path to the galaxy image file:\n")

outmap = image.rsplit(".", 1)[0] + "_mask.fits"  # output name


with fits.open(image) as hdulist:
    img = hdulist[0].data  # image data
    w = WCS(image)

verts = np.genfromtxt(drawing, skip_header=1, usecols=(2,3,4,5))  # vertices
if verts[:,2:].all() == 0:
    verts = w.all_world2pix(verts[:,:2], 1)  # converts world to pixel
    grid = func.imgmask(img, verts)
else:
    verts = verts[:,2:]
    grid = func.imgmask(img, verts)

fits.writeto(outmap, grid, overwrite=True)
