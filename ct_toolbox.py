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
	spaceY = np.linspace(-1, 1, resY)
	
	coordX = np.repeat(spaceX[:, np.newaxis], resY, axis=1)
	coordY = np.repeat(spaceY[np.newaxis, :], resX, axis=0)
	
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
	numDetectors = sinogram.shape[0]
	numProjections = sinogram.shape[1]
	result = np.zeros([numDetectors, numDetectors])
	angleRange = np.linspace(0, math.pi, num=numProjections, endpoint=False)
	for i in range(numProjections):
		backprojection = np.repeat(sinogram[:, i, np.newaxis].T, numDetectors, axis=0)
		result = result + skimage.transform.rotate(backprojection, math.degrees(angleRange[i]), clip=False, mode='constant', cval=0)
		#rotatedProj = skimage.transform.rotate(backprojection, math.degrees(angleRange[i]), mode='constant', cval=0)
		#plt.imshow(result.T, origin='lower', cmap='gray')
		#plt.colorbar()
		#plt.show()
	return result
	
def filterSinogram(sinogram):
	numDetectors = sinogram.shape[0]
	numProjections = sinogram.shape[1]
	fftSinogram = np.fft.fft(sinogram, axis=0)
	filter = np.abs(np.fft.fftfreq(numDetectors))*2
	filteredFftSinogram = fftSinogram * filter[:, np.newaxis]
	return np.real(np.fft.ifft(filteredFftSinogram, axis=0))
	
def makeForwardMatrix(numDetectors, numProjections, resX, resY):
	angleRange = np.linspace(0, math.pi, num=numProjections, endpoint=False)
	result = np.zeros([numProjections*numDetectors, resX*resY])
	for i in range(numProjections):
		for j in range(numDetectors):
			line = np.zeros(numDetectors)
			segmentlength = 2  #length of the line segment we integrate over
			line[j]=segmentlength*(numDetectors/resY)/resX
			affectedArea = skimage.transform.resize(line[np.newaxis, :], (resX, resY))
			#mode=constant, cval=0 specifies that when sampling outside of the image extents the image is assumed to be zero
			#this isn't entirely accurate outside of the scanning circle
			#but it shouldn't matter too much because all values outside the scanning circle are 0 in our experiments
			#however I've seen that the mode setting does have some effect on the artifacts visible in the reconstruction
			rotatedArea = skimage.transform.rotate(affectedArea, math.degrees(angleRange[i]), mode='constant', cval=0)
			#plt.imshow(rotatedArea.T, origin='lower', cmap='gray')
			#plt.colorbar()
			#plt.show()
			rotatedArea.shape = (result.shape[1])
			result[i+j*numProjections, :] = rotatedArea
	return result