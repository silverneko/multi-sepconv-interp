from sys import argv
import os
import numpy as np
import skimage
import skimage.io
import skimage.util
from skimage.filters.rank import entropy
from skimage.morphology import disk
import subprocess
import progressbar
import warnings

thres_1 = 3
thres_2 = 0.5
if len(argv) < 2:
    print ('Usage: {} [filelist] [ent_threshold={}] [ratio_threshold={}]'
           .format(argv[0], thres_1, thres_2))
    print ('ent_threshold: number of entropy bits to consider it low texture')
    print ('ratio_threshold: if ratio of low texture is larger then threshold'
           ' then discard the sample')
    exit(1)

filelist = argv[1]
if len(argv) > 2:
    thres_1 = float(argv[2])
if len(argv) > 3:
    thres_2 = float(argv[3])

with open(filelist, 'r') as fd:
    filenames = [f.rstrip() for f in fd.readlines()]

pbar = progressbar.ProgressBar()
ent = None
for f in pbar(filenames):
    v0 = skimage.io.imread(f+'_0.png')
    v0 = skimage.color.rgb2gray(v0)
    v0 = skimage.util.img_as_ubyte(v0)

    # There's no need to allocate a new out array every iteration
    if ent is None:
        ent = np.empty_like(v0)

    entropy(v0, disk(5), out=ent)

    if (ent < thres_1).sum() / ent.size > thres_2:
        # Do nothing
        pass
    else:
        print(f)
