import math
import numpy as np
from matplotlib import pyplot as plt
import ct_toolbox as ctt
import example_utilities as exutil


phantom = exutil.makeThreeCirclePhantom()

resX = 50
resY = resX
numDetectors = 50
numProjections = 50

groundTruth = ctt.rasterize(phantom, resX, resY)
groundTruth.shape = (resX*resY)
sinogram = ctt.makeSinogram(phantom, numDetectors, numProjections)
forwardMatrix = ctt.makeForwardMatrix(numDetectors, numProjections, resX, resY)
altSinogram = np.dot(forwardMatrix, groundTruth)
altSinogram.shape = (numDetectors, numProjections)

plt.subplot(121)
plt.imshow(altSinogram.T, origin='lower', cmap='gray')
plt.title("Forward Matrix")
plt.colorbar()
plt.subplot(122)
plt.imshow(sinogram.T, origin='lower', cmap='gray')
plt.title("Ray Tracing")
plt.colorbar()
plt.suptitle("Sinogram Comparison")
plt.show()

groundTruth.shape = (resX, resY)
sinogram.shape = (numDetectors*numProjections)
#rcond=0.075 means that all singular values smaller then 0.075 times the largest singular value are discarded
#Some regularization is required to get a good solution. With rcond=None you see only noise
reconstruction, _, rank, _ = np.linalg.lstsq(forwardMatrix, sinogram, rcond=0.075)
reconstruction.shape = (resX, resY)
print("Used singular values: {}".format(rank))

exutil.plotReconstructionComparison(reconstruction, groundTruth, "Least Squares Reconstruction")