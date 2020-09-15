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

resX = 50
resY = resX
numDetectors = 50
numProjections = 50

groundTruth = ctt.rasterize(phantom, resX, resY)
sinogram = ctt.makeSinogram(phantom, numDetectors, numProjections)
altSinogram = ctt.applyForwardMatrixIteratively(groundTruth, numDetectors, numProjections)

plt.subplot(121)
plt.imshow(altSinogram.T, origin='lower', cmap='gray')
plt.title("Iteratively Applied Forward Matrix")
plt.colorbar()
plt.subplot(122)
plt.imshow(sinogram.T, origin='lower', cmap='gray')
plt.title("Ray Tracing")
plt.colorbar()
plt.suptitle("Sinogram Comparison")
plt.show()

result = ctt.artReconstruction(sinogram, resX, resY, updateThreshold=0.01)
exutil.plotReconstructionComparison(result, groundTruth, "ART on simulated sinogram")