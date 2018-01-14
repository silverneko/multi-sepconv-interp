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

data = [(name, float(flowmag))
        for f in filenames
        for [name, flowmag] in (f.split(), )
        ]

def inRange(a, lb, rb):
    return lb <= a and a < rb

bin1 = list(filter(lambda a: inRange(a[1], 1, 10), data))
res = list(filter(lambda a: a[1] >= 10, data))

np.random.shuffle(bin1)

clipN = int(len(res) / 3)

bin1 = bin1[:clipN]

newdata = bin1 + res
newdata.sort(key=lambda a: a[1])

for name, flow in newdata:
    print ('{} {}'.format(name, flow))
