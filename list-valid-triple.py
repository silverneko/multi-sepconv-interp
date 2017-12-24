from sys import argv
import os
import numpy as np
import skimage
import skimage.io
import pyflow.pyflow as pyflow
import subprocess
import progressbar
import warnings

if len(argv) < 2:
    print ('Usage: {} [datadir]'.format(argv[0]))
    exit(1)

datadir = argv[1]

filenames = os.listdir(datadir)
valid = [os.path.join(datadir, f[:-6]) for f in filenames if f[-6:] == '_2.png']

for f in valid:
    print(f)
