import math
import numpy as np
from matplotlib import pyplot as plt
import skimage.transform
import ct_toolbox as ctt
import example_utilities as exutil

phantom = exutil.makeThreeCirclePhantom()

resX = 50
resY = resX
numDetectors = 50
numProjections = 50

groundTruth = ctt.rasterize(phantom, resX, resY)
sinogram = ctt.makeSinogram(phantom, numDetectors, numProjections)

result = ctt.sartReconstruction(sinogram, resX, resY, maxIterations=3)
exutil.plotReconstructionComparison(result, groundTruth, "SART on simulated sinogram")