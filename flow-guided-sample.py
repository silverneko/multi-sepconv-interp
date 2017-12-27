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

clipN = len(list(filter(lambda a: inRange(a[1], 4, 5), data)))

bin1 = list(filter(lambda a: a[1] < 1, data))
bin2 = list(filter(lambda a: inRange(a[1], 1, 2), data))
bin3 = list(filter(lambda a: inRange(a[1], 2, 3), data))
bin4 = list(filter(lambda a: inRange(a[1], 3, 4), data))
res = list(filter(lambda a: a[1] >= 4, data))

np.random.shuffle(bin1)
np.random.shuffle(bin2)
np.random.shuffle(bin3)
np.random.shuffle(bin4)

bin1 = bin1[:clipN]
bin2 = bin2[:clipN]
bin3 = bin3[:clipN]
bin4 = bin4[:clipN]

newdata = bin1 + bin2 + bin3 + bin4 + res
newdata.sort(key=lambda a: a[1])

for name, flow in newdata:
    print ('{} {}'.format(name, flow))
