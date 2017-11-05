"""
This script calculates the point-spread function (PSF) of a FITS image using
stars as point sources. The user is prompted to enter the path of the input.
It then proceeds as follows:
    1.  locates the stars and crops a large annulus around them
    2.  models the stellar profiles
    3.  performs Gaussian fitting and locates centroid
    3.  centres cut-out on new centroid
    4.  stacks the images of all stars and yields a global PSF.
"""


import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.nddata import PartialOverlapError, NoOverlapError
import func


## INPUT ##
fitme = input("Enter the path to the input file:\n")

varlist = []
for line in open(fitme):  # reads in file
    if not line.startswith(("\n", "#")):  # disregards newlines and comments
        val = [x.strip() for x in line.split(")")][1]  # extracts input value
        varlist.append(val)  # appends to variable list

inmap, incoord, outmap, nmap, outrad, cutsize, satur = varlist  # unpacking

if outmap == "NONE":  # defines PSF output filename
    filename = inmap.split(".fits")[0]
    outmap = filename + "_PSF.fits"

if satur == "NONE":  # deals with saturation value
    satur = np.inf
    print("\nSaturation value is set to infinity.")

outrad = float(outrad)
cutsize = int(cutsize)
starcoords = np.loadtxt(incoord)  # (x,y) coords of stars


## SKYMAP MANIPULATION ##
hdulist = fits.open(inmap)  # loads fits file
img = hdulist[0].data  # image data
w = WCS(hdulist[0].header)  # WCS information

PSF = np.zeros((cutsize, cutsize))  # PSF array
skipped = 0  # oversaturated stars are skipped
for coord in starcoords:
    try:
        # creates cutout object from original image
        star = Cutout2D(img, coord, 2*outrad*u.arcsec, wcs=w, mode="strict")
    except (PartialOverlapError, NoOverlapError) as e:
        print(e)

    star = star.data  # extracts data from cutout
    maxpos = np.unravel_index(np.nanargmax(star), star.shape)  # coords of peak

    star = star[
                maxpos[0]-cutsize//2 : maxpos[0]+cutsize//2 + 1,
                maxpos[1]-cutsize//2 : maxpos[1]+cutsize//2 + 1
                ]  # crops a box around the coords of the peak

    if not np.max(star) == satur:  # checks for saturated pixels
        centroid = func.fitgaussian(star)[1:3].round()  # centroid (x,y) coords
        # vector to new centroid
        coord_diff = centroid - np.repeat(cutsize//2, 2)

        if coord_diff.any():  # checks if centroid is different to guess
            upd_coord = coord + coord_diff  # new centroid
            # creates new cutour object from original image
            star = Cutout2D(img, upd_coord, 2*outrad*u.arcsec, wcs=w, mode="strict")
            star = star.data  # extracts data from cutout
            star = star[
                maxpos[0]-cutsize//2 : maxpos[0]+cutsize//2 + 1,
                maxpos[1]-cutsize//2 : maxpos[1]+cutsize//2 + 1
                ]  # crops a box around the coords of the peak
    else:
        skipped += 1

    PSF += star  # stacks star onto PSF

PSF /= len(starcoords) - skipped

# SAVE, SEE HOW GALFIT WORKS FIRST
