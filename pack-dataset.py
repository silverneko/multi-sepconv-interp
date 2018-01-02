import shutil
import os
import progressbar
from sys import argv

if len(argv) < 3:
    print ('Usage: {} [filelist] [outputdir]'.format(argv[0]))
    exit(1)

filelist = argv[1]
with open(filelist, 'r') as fd:
    filenames = [f.rstrip() for f in fd.readlines()]

outputdir = argv[2]
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

pbar = progressbar.ProgressBar()
for i, line in pbar(list(enumerate(filenames))):
    [name, flowmag] = line.split()
    outname = 'v_{}'.format(str(i).rjust(6, '0'))
    for suffix in ['_0.png', '_1.png', '_2.png']:
        shutil.copy(name+suffix, os.path.join(outputdir, outname+suffix))

    print ('{} {}'.format(os.path.join(outputdir, outname), flowmag))
