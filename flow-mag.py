from sys import argv
import os
import numpy as np
import skimage
import skimage.io
import pyflow.pyflow as pyflow

if len(argv) < 3:
    print ('Usage: {} [image1] [image2]'.format(argv[0]))
    exit(1)

pyflow_alpha = 1        # Regulization weight
pyflow_ratio = 0.75     # Downsampling ratio
pyflow_minWidth = 3    # Width of the coarsest level
pyflow_nOuterFPIterations = 5   # Number of outer fixed point iterations
pyflow_nInnerFPIterations = 1   # Number of inner fixed point iterations
pyflow_nSORIterations = 30  # Number of SOR iterations
pyflow_colType = 0      # 0: RGB; 1: Grayscale

image1 = skimage.io.imread(argv[1])
image2 = skimage.io.imread(argv[2])

h, w = image1.shape[:2]

u, v, im2W = pyflow.coarse2fine_flow(
    skimage.util.img_as_float(image1),
    skimage.util.img_as_float(image2),
    pyflow_alpha,
    pyflow_ratio,
    pyflow_minWidth,
    pyflow_nOuterFPIterations,
    pyflow_nInnerFPIterations,
    pyflow_nSORIterations,
    pyflow_colType
)

mag = np.sum(np.sqrt(u * u + v * v))
meanMag = mag / (h * w)

print ('Mag: {} ; MeanMag: {}'.format(mag, meanMag))
