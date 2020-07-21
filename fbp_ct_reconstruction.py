import math
import numpy as np
import skimage.transform
from matplotlib import pyplot as plt

class Circle:
	def __init__(self, center, radius, attenuation):
		self.center = center
		self.radius = radius
		self.squaredRadius = radius ** 2
		self.attenuation = attenuation
	
	def __repr__(self):
		return "Circle( {} , {}, {})".format(self.center, self.radius, self.attenuation)


def makeCoordsMatrix(resX, resY):
	spaceX = np.linspace(-1, 1, resX)
	spaceY = np.linspace(-1, 1, resX)
	
	coordX = np.repeat(spaceX[:, np.newaxis], resY, axis=1)
	coordY = np.repeat(spaceY[np.newaxis, :], resY, axis=0)
	
	return np.stack((coordX, coordY), axis=-1)

def rasterize(circles, resX, resY):
	image = np.zeros([resX, resY])
	coords = makeCoordsMatrix(resX, resY)
	for circle in circles:
		withinRadius = ((coords[:, :, 0]-circle.center[0])**2  + 
		                (coords[:, :, 1]-circle.center[1])**2) < circle.squaredRadius
		image = image + withinRadius * circle.attenuation
	
	return image


# Currently only non-overlapping circles are supported in the intersection test
def projectRay(circles, perpDir, offset):
	attenuation = 1
	for circle in circles:
		d = np.dot(perpDir, circle.center) - offset
		d2 = d**2
		if d2<circle.squaredRadius:
			l = 2*math.sqrt(circle.squaredRadius - d2)
			attenuation = attenuation * math.exp(-circle.attenuation*l)
	return attenuation

def makeProjection(circles, angle, resolution):
	result = np.zeros(resolution)
	offsetRange = np.linspace(-1, 1, resolution)
	for i in range(resolution):
		result[i] = projectRay(circles, np.array([-math.sin(angle), math.cos(angle)]), offsetRange[i])
	return result

def makeSinogram(circles, numDetectors, numProjections):
	sinogram = np.zeros([numDetectors, numProjections])
	angleRange = np.linspace(0, math.pi, num=numProjections, endpoint=False)
	for i in range(numProjections):
		sinogram[:, i] = makeProjection(circles, angleRange[i], numDetectors)
	return -np.log(sinogram)


def makeBackprojection(sinogram):
	numProjections = sinogram.shape[1]
	numDetectors = sinogram.shape[0]
	result = np.zeros([numDetectors, numDetectors])
	angleRange = np.linspace(0, math.pi, num=numProjections, endpoint=False)
	for i in range(numProjections):
		backprojection = np.repeat(sinogram[:, i, np.newaxis].T, numDetectors, axis=0)
		result = result + skimage.transform.rotate(backprojection, math.degrees(angleRange[i]), mode='constant', cval=0)
	return result
	
def filterSinogram(sinogram):
	numProjections = sinogram.shape[1]
	numDetectors = sinogram.shape[0]
	fftSinogram = np.fft.fft(sinogram, axis=0)
	filter = np.abs(np.fft.fftfreq(numDetectors))*2
	filteredFftSinogram = fftSinogram * filter[:, np.newaxis]
	return np.real(np.fft.ifft(filteredFftSinogram, axis=0))

phantom = [Circle(np.array([-0.5, 0.5]), 0.2, 1),
           Circle(np.array([-0.4, -0.4]), 0.3, 2),
           Circle(np.array([0.5, 0.1]), 0.4, 3)]

proj = makeProjection(phantom, 0, 20)
print(proj)

sinogram = makeSinogram(phantom, 300, 300)
filteredSinogram = filterSinogram(sinogram)

plt.subplot(121)
plt.imshow(sinogram.T, origin='lower', cmap='gray')
plt.title("Sinogram")
plt.colorbar()
plt.subplot(122)
plt.imshow(filteredSinogram.T, origin='lower', cmap='gray')
plt.title("Filtered Sinogram")
plt.colorbar()
plt.show()

reconstruction = makeBackprojection(filteredSinogram)
groundTruth = rasterize(phantom, 300, 300)
error = reconstruction-groundTruth
vmin = min(np.min(reconstruction), np.min(groundTruth))
vmax = max(np.max(reconstruction), np.max(groundTruth))

plt.subplot(131)
plt.imshow(reconstruction.T, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
plt.title("Reconstruction")
plt.colorbar()

plt.subplot(132)
plt.imshow(groundTruth.T, origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
plt.title("Original")
plt.colorbar()

plt.subplot(133)
plt.imshow(error.T, origin='lower')
plt.title("Error")
plt.colorbar()

plt.show()