"""
This script calculates the point-spread function (PSF) of a FITS image using
stars as point sources. The user is prompted to enter the path of the input.
The input should be a file with information about the map and some approximate
stellar coordinates on the map. Other parameters may also be used.
It then proceeds as follows:
    1.  locates the stars and crops a large annulus around them
    2.  models the stellar profiles
    3.  performs Gaussian fitting and locates centroid
    3.  centres cut-out on new centroid
    4.  stacks the images of all stars and yields a global PSF.
"""


import re
import numpy as np
from scipy.ndimage import shift
from scipy.stats import norm as gauss
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.nddata import PartialOverlapError, NoOverlapError
import func


## INPUT ##
fitme = input("Enter the path to the input file:\n")

varlist = []
with open(fitme) as f:
    for line in f:  # reads in file
        if not line.startswith(("\n", "#", "=")):  # disregards comments
            val = re.split(" ", line)[1]  # extracts parameter
            varlist.append(val)  # appends to varlist

# unpacking
inmap, incoord, outmap, nmap, outrad, cutsize, satur, clipval = varlist

# defines output filename of SCI with ADU flux
ADUmap = re.split(" |_", inmap)[0] + "_ADU.fits"
# defines PSF and noisemap output filenames
if outmap == "none": outmap = re.split(" |_", inmap)[0] + "_psf.fits"
if nmap == "none": nmap = re.split(" |_", inmap)[0] + "_noisemap.fits"
# deals with saturation value
if satur == "none": satur = np.inf; print("Saturation value set to infinity.")

outrad, satur, clipval = map(float, [outrad, satur, clipval])
cutsize = int(cutsize)
starcoords = np.loadtxt(incoord, skiprows=6)  # (x,y) coords of stars
starcoords = np.reshape(starcoords, (-1, 2))  # ensures 2d array for retrieval


## FILE IMPORT ##
with fits.open(inmap) as hdulist:
    img = hdulist[0].data  # image data
    hdr = hdulist[0].header  # image header
    w = WCS(hdulist[0].header)  # WCS object
    img /= hdr["CCDGAIN"]  # converts e- to ADU


## NOISEMAP EXTRACTION ##
SD = clipval/gauss.interval(0.5)[1]  # IQR to SD conversion factor
SDbounds = SD*np.nanpercentile(img, [25,75])  # +/- 1 SD bounds
noise = img[(img > SDbounds[0]) & (img < SDbounds[1])].std()  # sky noise
NOISE = np.sqrt(noise**2 + img)  # total noise


## SKYMAP MANIPULATION ##
PSF = np.zeros((cutsize, cutsize))  # PSF array
cmax = 0  # maximum offset from centre
for coord in starcoords:
    try:
        # creates cutout object from original image
        star = Cutout2D(img, coord, 2*outrad*u.arcsec, wcs=w, mode="strict")
    except (PartialOverlapError, NoOverlapError) as e:
        print(e)

    star = star.data  # extracts data from cutout
    maxpos = np.unravel_index(np.nanargmax(star), star.shape)  # coords of peak

    # crops a box around the coords of the peak
    star = star[
                maxpos[0]-cutsize//2 : maxpos[0]+cutsize//2 + 1,
                maxpos[1]-cutsize//2 : maxpos[1]+cutsize//2 + 1
                ]

    if not star.max() == satur:  # checks for saturated pixels
        centroid = func.fitgaussian(star)[1:3] + 1  # centroid (x,y) coords
        # vector to new centroid (rows increase downwards)
        cdiff = np.repeat((cutsize+1)//2, 2) - centroid
        star = shift(star, cdiff)  # shifts to new centre
        PSF += star  # stacks star onto PSF

        # stores maximum pixel offset
        cmax = np.max((cmax, np.abs(cdiff).max()))

cmax = int(np.ceil(cmax))
PSF = PSF[cmax:-cmax, cmax:-cmax]  # crops out shifted edges


## OUTPUT ##
fits.writeto(outmap, PSF, overwrite=True)  # exports PSF
fits.writeto(ADUmap, img, hdr, overwrite=True)  # exports fits in ADU
fits.writeto(nmap, NOISE, hdr, overwrite=True)  # exports noisemap
