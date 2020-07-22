import math
import numpy as np
import skimage.transform
from matplotlib import pyplot as plt
import ct_toolbox as ctt

#All circles must be fully contained within a circle with radius 1 and origin 0.0
#because that is assumed during the reconstructions
phantom = [ctt.Circle(np.array([-0.5, 0.5]), 0.2, 1),
           ctt.Circle(np.array([-0.4, -0.4]), 0.3, 2),
           ctt.Circle(np.array([0.5, 0.1]), 0.4, 3)]

resolution = 512

sinogram = ctt.makeSinogram(phantom, resolution, resolution)
filteredSinogram = ctt.filterSinogram(sinogram)

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
#reconstruction = skimage.transform.iradon(sinogram, np.linspace(0, 180, resolution, endpoint=False), resolution)*resolution/2
groundTruth = ctt.rasterize(phantom, resolution, resolution)
error = reconstruction-groundTruth
vmin = min(np.min(reconstruction), np.min(groundTruth))
vmax = max(np.max(reconstruction), np.max(groundTruth))

plt.subplot(131)
plt.imshow(reconstruction.T, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
plt.title("FBP Reconstruction")
plt.colorbar()

plt.subplot(132)
plt.imshow(groundTruth.T, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
plt.title("Rasterized Phantom")
plt.colorbar()

plt.subplot(133)
plt.imshow(error.T, origin='lower', cmap='PuOr_r', vmin=-1.5, vmax=1.5)
plt.title("Error")
plt.colorbar()

plt.show()
