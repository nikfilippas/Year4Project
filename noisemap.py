"""
This script outputs the noisemap of an input FITS file.
Input must be the SCI image of which the noise map is to be calculated.
"""

import re
import numpy as np
from scipy.stats import norm as gauss
from astropy.io import fits


## INPUT & PARAMETERS ##
inmap = input("Enter the path to the input file:\n")  # path to input file
nmap = re.split(" |_", inmap)[0] + "_noise.fits"  # output filename

clipval = 5  # fits image sigma-clipping cutoff  [SD]

## ANALYSIS ##
SD = clipval/gauss.interval(0.5)[1]  # IQR to SD conversion factor

hdulist = fits.open(inmap)  # loads fits file
img = hdulist[0].data  # image data
img /= hdulist[0].header["CCDGAIN"]  # converts e- to ADU

SDbounds = SD*np.nanpercentile(img, [25,75])  # +/- 1 SD bounds
noise = img[(img > SDbounds[0]) & (img < SDbounds[1])].std()  # sky noise
NOISE = np.sqrt(noise**2 + img)  # total noise

## OUTPUT ##
fits.writeto(nmap, NOISE, hdulist[0].header, overwrite=True)
