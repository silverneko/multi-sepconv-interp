import argparse
import os
import time
import numpy as np
import progressbar
import skimage.io
import skimage.util
import torch
from Sepconv import Sepconv
from Sepconv2 import Sepconv2

parser = argparse.ArgumentParser(
    description='eval.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--dim', default=1, type=int, metavar='D',
                    help='model estimates filter of dimension D')
parser.add_argument('--model', type=str, metavar='MODEL',
                    help='model parameter')
parser.add_argument('--first', type=str, metavar='FIRST')
parser.add_argument('--second', type=str, metavar='SECOND')
parser.add_argument('-o', '--output', default='out.png', type=str,
                    metavar='OUTPUT',
                    help='output file name')
parser.add_argument('--cpu', action='store_true', default=False,
                    help="don't use cuda")
parser.add_argument('--model-type', default='v1', type=str, metavar='TYPE',
                    help='use model v1 or v2')

def eval(model, img0_name, img1_name):
    img0 = skimage.io.imread(img0_name)
    img1 = skimage.io.imread(img1_name)

    img0 = skimage.util.img_as_float32(img0)
    img1 = skimage.util.img_as_float32(img1)

    if len(img0.shape) == 2:
        img0 = np.stack([img0, img0, img0], axis=-1)
        img1 = np.stack([img1, img1, img1], axis=-1)

    h, w, c = img0.shape
    newh = int(np.ceil(h / 32)) * 32
    neww = int(np.ceil(w / 32)) * 32

    padu = int((newh - h) / 2)
    padd = newh - h - padu
    padl = int((neww - w) / 2)
    padr = neww - w - padl

    img0 = np.pad(img0, ((padu, padd), (padl, padr), (0,0)), 'edge')
    img1 = np.pad(img1, ((padu, padd), (padl, padr), (0,0)), 'edge')

    img0 = np.ascontiguousarray(img0)
    img1 = np.ascontiguousarray(img1)

    img0 = torch.from_numpy(img0.transpose( (2, 0, 1) ))
    img1 = torch.from_numpy(img1.transpose( (2, 0, 1) ))

    img0 = img0.unsqueeze(0)
    img1 = img1.unsqueeze(0)

    if not args.cpu:
        img0 = img0.cuda()
        img1 = img1.cuda()

    img0 = torch.autograd.Variable(img0, volatile=True)
    img1 = torch.autograd.Variable(img1, volatile=True)

    model.eval()
    res = model(img0, img1)
    res = res.clamp(min=0.0, max=1.0)

    res = res.cpu().data.numpy().squeeze(0)
    res = res.transpose((1, 2, 0))

    res = res[padu:padu+h, padl:padl+w]

    return res

def main():
    global args
    args = parser.parse_args()

    if args.model_type == 'v1':
        model = Sepconv(args.dim)
    elif args.model_type == 'v2':
        model = Sepconv2(args.dim)
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model'])

    if not args.cpu:
        model = model.cuda()

    if args.first:
        res = eval(model, args.first, args.second)
        skimage.io.imsave(args.output, res)
        return

    def loss(x, y):
        assert(x.shape == y.shape)
        return np.abs(x - y).mean()

    dirs = """
Army
Backyard
Basketball
Dumptruck
Evergreen
Grove
Mequon
Schefflera
Urban
Wooden
""".split()
    dirs = [os.path.join('eval-data', d) for d in dirs]

    L = []
    for d in dirs:
        Ld = []
        for i in range(7, 13):
            tri = [os.path.join(d, 'frame{:02}.png'.format(t)) for t in range(i, i+3)]
            y = eval(model, tri[0], tri[2])
            ground_truth = skimage.io.imread(tri[1])
            ground_truth = skimage.util.img_as_float32(ground_truth)
            if len(ground_truth.shape) == 2:
                y = y[:,:,0]
            Ld.append(loss(ground_truth, y))
        L += Ld
        print (d)
        print (Ld)
        print (sum(Ld) / len(Ld))

    print (sum(L) / len(L))

if __name__ == '__main__':
    main()
