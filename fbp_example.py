import math
import numpy as np
from matplotlib import pyplot as plt
import ct_toolbox as ctt
import example_utilities as exutil

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

exutil.plotReconstructionComparison(reconstruction, rasterizedPhantom, "FBP Reconstruction")
