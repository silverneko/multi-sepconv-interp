from sys import argv
import os
import numpy as np
import skimage
import skimage.io
import pyflow.pyflow as pyflow
import subprocess
import progressbar
import warnings

ffmpeg_thread_num = 4

def get_video_info(f):
    command = [
        'ffprobe',
        '-v', 'fatal',
        '-show_entries', 'stream=width,height,r_frame_rate,duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        f
    ]
    p = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE)
    out, err = p.communicate()

    if err:
        print (err)
        return ((), True)

    out = out.decode().split('\n')
    try:
        vinfo = {
            'file' : f,
            'width': int(out[0]),
            'height' : int(out[1]),
            'fps': float(out[2].split('/')[0])/float(out[2].split('/')[1]),
            'duration' : float(out[3])
        }
    except:
        print ('Exception while reading video info:', f, out)
        return ((), True)
    return (vinfo, False)

def get_video_frame(f, width, height, t, nframes):
    command = [
        'ffmpeg',
        '-loglevel', 'fatal',
        '-ss', str(t),
        '-i', f,
        '-threads', str(ffmpeg_thread_num),
        '-frames:v', str(nframes),
        '-vsync', '0',
        '-f', 'image2pipe',
        '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo',
        '-'
    ]
    p = subprocess.Popen(command, stderr=subprocess.PIPE ,stdout = subprocess.PIPE)
    out, err = p.communicate()
    if err:
        print (err)
        return ((), True)

    try:
        frames = np.fromstring(out, dtype='uint8').reshape(nframes, height, width, 3)
    except:
        print ('Exception while extracting frame:', f)
        return ((), True)
    return (frames, False)


if len(argv) < 4:
    print ('Usage: {} [datadir] [#sample_per_video] [outputdir] [resume from] [end at]'.format(argv[0]))
    exit(1)

datadir = argv[1]
nsamples_per_video = int(argv[2])
outputdir = argv[3]

resume_from = 0
if len(argv) > 4:
    resume_from = int(argv[4])

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

#np.random.seed(0xDEADBEEF)
patchsize = 150

filenames = os.listdir(datadir)
pathnames = [os.path.join(datadir, f) for f in filenames]

sample_count = 0

filelist = list(enumerate(pathnames))
nfiles = len(filelist)
end_at = nfiles
if len(argv) > 5:
    end_at = min(end_at, int(argv[5]))

filelist = filelist[resume_from:end_at]
pbar = progressbar.ProgressBar()
pbar.start(end_at)
for i, f in filelist:
    pbar.update(i)
    vinfo, err = get_video_info(f)
    if err:
        continue

    if vinfo['duration'] < 1:
        print ('Video too short: {}'.format(f))
        continue

    w = vinfo['width']
    h = vinfo['height']

    for j in range(nsamples_per_video):
        t = np.random.random() * (vinfo['duration'] - 1)
        triple, err = get_video_frame(f, w, h, t, 3)
        if err:
            continue

        sample_row = int(np.random.random() * (h - patchsize))
        sample_col = int(np.random.random() * (w - patchsize))

        patch = [
            frame[sample_row:sample_row+patchsize,
                  sample_col:sample_col+patchsize]
            for frame in triple
        ]

        outname = 'v_{}_p_{}'.format(str(i).rjust(4, '0'), str(j).rjust(3, '0'))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(os.path.join(outputdir, outname + '_0.png'), patch[0])
            skimage.io.imsave(os.path.join(outputdir, outname + '_1.png'), patch[1])
            skimage.io.imsave(os.path.join(outputdir, outname + '_2.png'), patch[2])
        sample_count += 1

pbar.finish()

print ('Sampled {} frame triples'.format(sample_count))
