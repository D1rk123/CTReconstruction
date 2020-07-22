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

resX = 100
resY = resX
numDetectors = 100
numProjections = 100

groundTruth = ctt.rasterize(phantom, resX, resY)
groundTruth.shape = (resX*resY)
sinogram = ctt.makeSinogram(phantom, numDetectors, numProjections)
forwardMatrix = ctt.makeForwardMatrix(numDetectors, numProjections, resX, resY)
altSinogram = np.dot(forwardMatrix, groundTruth)
altSinogram.shape = (numDetectors, numProjections)

plt.subplot(121)
plt.imshow(sinogram.T, origin='lower', cmap='gray')
plt.title("Simulated sinogram")
plt.colorbar()
plt.subplot(122)
plt.imshow(altSinogram.T, origin='lower', cmap='gray')
plt.title("Sinogram calculated by forward matrix")
plt.colorbar()
plt.show()

groundTruth.shape = (resX, resY)
sinogram.shape = (numDetectors*numProjections)
#Some regularization is required to get a good solution. With rcond=None you only see noise
reconstruction, _, rank, _ = np.linalg.lstsq(forwardMatrix, sinogram, rcond=0.075)
reconstruction.shape = (resX, resY)
print("Used singular values: {}".format(rank))

plt.subplot(121)
plt.imshow(groundTruth.T, origin='lower', cmap='gray')
plt.title("groundTruth")
plt.colorbar()
plt.subplot(122)
plt.imshow(reconstruction.T, origin='lower', cmap='gray')
plt.title("reconstruction")
plt.colorbar()
plt.show()