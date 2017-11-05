"""
This script creates a subcatalogue of GAMA_DR1, extracting the following:
1.  The RA and Dec of each source
2.  The 500 um flux of the source and its associated error
Note: Only sources with a flux > 70 mJy are considered.
"""

from __future__ import division
import numpy as np

# RA, Dec, F500, E500
hatlas = np.genfromtxt("../cats/HATLAS_DR1_CATALOGUE_V1.2.DAT",
									skip_header=39, usecols=(3,4,7,10))

hatlas[:,2:] *= 1000  # converts to [mJy]
hatlas = hatlas[hatlas[:,2] >= 70]  # cutoff is [70 mJy]

hdr = "RA DEC F500 E500"
np.savetxt("../cats/hatlas.txt", hatlas, fmt="%.16f", header=hdr, comments="")
