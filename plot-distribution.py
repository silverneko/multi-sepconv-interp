import matplotlib.pyplot as plt
import numpy as np
import math
from sys import argv

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

data = [b for (a, b) in data]
lbound = int(math.ceil(max(data)))

plt.hist(data, bins=np.arange(lbound, step=0.1), weights=np.ones(len(data)) / float(len(data)) * 100)

plt.xlim(xmin=0, xmax=15)
plt.title('Flow magnitude distribution')
plt.xlabel('Mean pixel flow magnitude')
plt.ylabel('Percentage')
plt.savefig('sampled_flow_dist_fine_scale.png', dpi=300)
