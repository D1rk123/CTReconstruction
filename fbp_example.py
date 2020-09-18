import math
import numpy as np
from matplotlib import pyplot as plt
import ct_toolbox as ctt
import example_utilities as exutil

phantom = exutil.makeThreeCirclePhantom()

resolution = 256

sinogram = ctt.makeSinogram(phantom, resolution, resolution)
filteredSinogram = ctt.filterSinogram(sinogram, False)

plt.subplot(121)
plt.imshow(sinogram.T, origin='lower', cmap='gray')
plt.title("Sinogram")
plt.colorbar()
plt.subplot(122)
plt.imshow(filteredSinogram.T, origin='lower', cmap='gray')
plt.title("Filtered Sinogram")
plt.colorbar()
plt.show()

reconstruction = ctt.makeBackprojection(filteredSinogram)
rasterizedPhantom = ctt.rasterize(phantom, resolution, resolution)

exutil.plotReconstructionComparison(reconstruction, rasterizedPhantom, "FBP Reconstruction")
