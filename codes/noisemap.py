"""
This script outputs the noisemap of an input FITS file.
"""

import numpy as np
from scipy.stats import norm as gauss
from astropy.io import fits


## INPUT & PARAMETERS ##
inmap = "../img/HATLASJ091331-003644_final_drz_sci.fits"  # path to input file
nmap = inmap.split(".fits")[0] + "_noise.fits"  # output filename

clipval = 5  # fits image sigma-clipping cutoff  [SD]

## ANALYSIS ##
SD = clipval/gauss.interval(0.5)[1]  # IQR to SD conversion factor

hdulist = fits.open(inmap)  # loads fits file
img = hdulist[0].data  # image data

SDbounds = SD*np.nanpercentile(img, [25,75])  # +/- 1 SD bounds
noise = img[(img > SDbounds[0]) & (img < SDbounds[1])].std()  # sky noise
NOISE = np.sqrt(noise**2 + img)  # total noise

## OUTPUT ##
fits.writeto(nmap, NOISE, hdulist[0].header, overwrite=True)
