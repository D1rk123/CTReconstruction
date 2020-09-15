import math
import numpy as np
from matplotlib import pyplot as plt

def plotReconstructionComparison(reconstruction, groundTruth, reconstructionTitle):
	vmin = min(np.min(reconstruction), np.min(groundTruth))
	vmax = max(np.max(reconstruction), np.max(groundTruth))
	error = reconstruction-groundTruth

	plt.subplot(131)
	plt.imshow(reconstruction.T, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
	plt.title(reconstructionTitle)
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