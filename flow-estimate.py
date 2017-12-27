from sys import argv
import os
import numpy as np
import skimage
import skimage.io
import pyflow.pyflow as pyflow
import progressbar

if len(argv) < 2:
    print ('Usage: {} [filelist]'.format(argv[0]))
    exit(1)

filelist = argv[1]
with open(filelist, 'r') as fd:
    filenames = [f.rstrip() for f in fd.readlines()]

pyflow_alpha = 1        # Regulization weight
pyflow_ratio = 0.75     # Downsampling ratio
pyflow_minWidth = 3    # Width of the coarsest level
pyflow_nOuterFPIterations = 5   # Number of outer fixed point iterations
pyflow_nInnerFPIterations = 1   # Number of inner fixed point iterations
pyflow_nSORIterations = 30  # Number of SOR iterations
pyflow_colType = 0      # 0: RGB; 1: Grayscale

pbar = progressbar.ProgressBar()
for f in pbar(filenames):
    v0 = skimage.io.imread(f+'_0.png')
    v1 = skimage.io.imread(f+'_2.png')

    u, v, im2W = pyflow.coarse2fine_flow(
        skimage.util.img_as_float(v0),
        skimage.util.img_as_float(v1),
        pyflow_alpha,
        pyflow_ratio,
        pyflow_minWidth,
        pyflow_nOuterFPIterations,
        pyflow_nInnerFPIterations,
        pyflow_nSORIterations,
        pyflow_colType
    )

    h, w = v0.shape[:2]
    mag = np.sqrt(u * u + v * v).sum()
    meanMag = mag / (h * w)

    print ('{} {}'.format(f, meanMag))
