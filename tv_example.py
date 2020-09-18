import math
import numpy as np
from matplotlib import pyplot as plt
import ct_toolbox as ctt
import example_utilities as exutil
import cvxpy as cp

phantom = exutil.makeThreeCirclePhantom()

resX = 50
resY = resX
numDetectors = 50
numProjections = 50

groundTruth = ctt.rasterize(phantom, resX, resY)
sinogram = ctt.makeSinogram(phantom, numDetectors, numProjections)
sinogram.shape = (numDetectors*numProjections)
forwardMatrix = ctt.makeForwardMatrix(numDetectors, numProjections, resX, resY)

reconstruction = cp.Variable(shape=(resX, resY))
k = 0.1
objective = cp.Minimize(
	  cp.sum_squares(forwardMatrix@cp.vec(reconstruction) - sinogram)
	+ k * cp.tv(reconstruction))
constraints = [0 <= reconstruction]
prob = cp.Problem(objective, constraints)
prob.solve(verbose=True, solver=cp.SCS)

# cvxpy uses column-major order for flattening matrices (cp.vec)
# while numpy uses row-major order
# therefore the reconstruction has to be transposed here
exutil.plotReconstructionComparison(reconstruction.value.T, groundTruth, "Least Squares Reconstruction")