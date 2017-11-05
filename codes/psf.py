"""
This script calculates the point-spread function (PSF) of a FITS image using
stars as point sources. The user is prompted to enter the path of the input.
It then proceeds as follows:
    1.  locates the stars and crops a large annulus around them
    2.  models the stellar profiles and restores over-saturated pixels
    3.  crops a set, smaller annulus around the stars
    4.  regrids all stars into one common size (scale factor is the PSF weight)
    5.  normalises the flux density
    6.  stacks the cropped images and normalise to produce the average PSF.
"""


import numpy as np
from scipy.stats import norm as gauss
from scipy.misc import imresize
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.nddata import PartialOverlapError, NoOverlapError
#import func


## INPUT ##
fitme = input("Enter the path to the input file:\n")


## PARAMETERS ##
clipval = 5  # fits iamge sigma-clipping cutoff  [SD]
cutsize = 51  # box cut-out size  [px]
outrad = 5  # box radius [arcsec]


## INPUT MANIPULATION ##
varlist = []
for line in open(fitme):  # reads in file
    if not line.startswith(("\n", "#")):  # disregards newlines and comments
        val = [x.strip() for x in line.split(")")][1]  # extracts input value
        varlist.append(val)  # appends to variable list

inmap, incoord, outmap, nmap, gaussext, totext, satur = varlist  # unpacking

if outmap == "NONE":  # defines PSF output filename
    filename = inmap.split(".fits")[0]
    outmap = filename + "_PSF.fits"

if nmap == "NONE":  # defines noisemap output filename
    filename = inmap.split(".fits")[0]
    nmap = filename + "_noise.fits"

starcoords = np.loadtxt(incoord)  # (x,y) coords of stars


## SKYMAP MANIPULATION ##
hdulist = fits.open(inmap)  # loads fits file
img = hdulist[0].data  # image data
w = WCS(hdulist[0].header)  # WCS information


for coord in starcoords:
    try:
        star = Cutout2D(
                img, coord,
                2*(outrad+2)*u.arcsec,
                wcs=w, mode="strict"
                )
    except (PartialOverlapError, NoOverlapError) as e:
        print(e)

    zoom = float(outrad+2)  # scale factor
    # regrids image
    rebin = imresize(star.data, zoom, interp="bicubic").astype(float)
    rebin /= zoom**2  # normalises for scale factor

    rebin /= rebin.sum()  # normalises for point source flux





## NOISE MAP ##
SD = clipval/gauss.interval(0.5)[1]  # IQR to SD conversion factor

SDbounds = SD*np.nanpercentile(img, [25,75])  # +/- 1 SD bounds
noise = img[(img > SDbounds[0]) & (img < SDbounds[1])].std()  # sky noise
NOISE = np.sqrt(noise**2 + img)  # total noise

fits.writeto(nmap, NOISE, hdulist[0].header, overwrite=True)

























