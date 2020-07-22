# CTReconstruction

This repository contains Python scripts for the simulation of CT data acquisition and for CT reconstruction. The ct_toolbox.py file contains all reusable functions and the fbp_example.py and least_squares_example.py files show how this functionality can be used to reconstruct a CT image in 2 different ways: matrix inversion and filtered back projection.

## CT Simulation
To simulate the acquisition of CT data the user can define a phantom consisting of nonoverlapping circles. From this continuous representation images can be generated at any resolution. By projecting the data stepwise in all directions a sinogram is created, where every row represents one projection. In this repository I always assume that parallel beams are used and that the source and detector are at a distance of 1 from the origin.

![Phantom and sinogram](https://raw.githubusercontent.com/D1rk123/CTReconstruction/master/GithubImages/PhantomAndSinogram.png)

## Reconstruction by Matrix Inversion
The process of calculating the projections from a given image can be modelled as a linear process: A**x** = **y**, where **x** is a vector representing the image and **y** is a vector representing the sinogram. In a typical reconstruction setting **y** is measured and A can be derived from the setup, so **x** can be calculated by solving the equation for **x**. In most of my experiments I chose **x** and **y** to be the same size and the corresponding matrix A turned out to be slightly rank deficient. However the equation can still be solved in the least squares sense. The code uses the singular value decomposition based solver from Numpy. It is configured to discard the lowest singular values to reduce the noise in the reconstructed image caused by a slight mismatch between matrix A and the simulation used for generating the sinogram.

![Matrix inversion reconstruction](https://raw.githubusercontent.com/D1rk123/CTReconstruction/master/GithubImages/LeastSquares.png)

Because matrix A has dimensions *all pixels in the reconstruction* x *all pixels in the sinogram* it can grow very quickly. That's also why I showed such a low resolution example. There are algorithms such as ART and SART that solve the same equation without needing the whole matrix to be in memory.

## Filtered Back Projection
Another technique to reconstruct a CT image is filtered back projection. In this technique the sinogram is filtered using a Ram Lak filter (high pass ramp filter) and every projection (row of the sinogram) is smeared back over the image in the same direction it was acquired.

![Filtered back projection reconstruction](https://raw.githubusercontent.com/D1rk123/CTReconstruction/master/GithubImages/FilteredBackProjection.png)

In the image above I used a straightforward ramp filter. As you can see the overall shape is reconstructed correctly, but most of the image is slightly underestimated. By looking at the implementation of filtered back projection in the Scikit-Image library I found out that they used a slightly smarter filter. I also made the filter from Scikit-Image available in my code. With that filter most of the image is correctly reconstructed.

![Filtered back projection reconstruction with smarter filter](https://raw.githubusercontent.com/D1rk123/CTReconstruction/master/GithubImages/FilteredBackProjectionSkimageFilter.png)

For more information on this filter you can have a look at [the Scikit-Image source code](https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/radon_transform.py#L184-L305) or at chapter 3 of the book they mentioned as a reference: http://www.slaney.org/pct/pct-toc.html.
