import math
import numpy as np
from matplotlib import pyplot as plt
import ct_toolbox as ctt

#All circles must be fully contained within a circle with radius 1 and origin 0.0
#because that is assumed during the reconstructions
phantom = [ctt.Circle(np.array([-0.5, 0.5]), 0.2, 1),
           ctt.Circle(np.array([-0.4, -0.4]), 0.3, 2),
           ctt.Circle(np.array([0.5, 0.1]), 0.4, 3)]

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
error = reconstruction-rasterizedPhantom
vmin = min(np.min(reconstruction), np.min(rasterizedPhantom))
vmax = max(np.max(reconstruction), np.max(rasterizedPhantom))

plt.subplot(131)
plt.imshow(reconstruction.T, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
plt.title("FBP Reconstruction")
plt.colorbar()

plt.subplot(132)
plt.imshow(rasterizedPhantom.T, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
plt.title("Rasterized Phantom")
plt.colorbar()

plt.subplot(133)
plt.imshow(error.T, origin='lower', cmap='PuOr_r', vmin=-1.5, vmax=1.5)
plt.title("Error")
plt.colorbar()

plt.show()

