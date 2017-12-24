from sys import argv
import os
import numpy as np
import skimage
import skimage.io
import pyflow.pyflow as pyflow
import subprocess
import progressbar
import warnings

def histDiff(h0, h1):
    return max([abs(x - y) for x, y in zip(h0, h1)])

threshold = 0.3
if len(argv) < 2:
    print ('Usage: {} [filelist] [threshold={}]'.format(argv[0], threshold))
    exit(1)

filelist = argv[1]
if len(argv) > 2:
    threshold = float(argv[2])

with open(filelist, 'r') as fd:
    filenames = [f.rstrip() for f in fd.readlines()]

pbar = progressbar.ProgressBar()
for f in pbar(filenames):
    v0 = skimage.io.imread(f+'_0.png')
    v1 = skimage.io.imread(f+'_2.png')
    v0 = skimage.color.rgb2gray(v0)
    v1 = skimage.color.rgb2gray(v1)
    hist0, _ = np.histogram(v0, bins=16, range=(0.0, 1.0))
    hist1, _ = np.histogram(v1, bins=16, range=(0.0, 1.0))
    hist0 = hist0 / v0.size
    hist1 = hist1 / v1.size
    diff = histDiff(hist0, hist1)
    if diff > threshold:
        # Do nothing
        pass
    else:
        print(f)
