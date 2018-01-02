import argparse
import os
import numpy as np
import progressbar
import skimage.io
import skimage.util
import torch
import torch.nn
import torch.optim
import torch.utils.data

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('imagelist', metavar='LIST',
                    help='list of image pathnames')
parser.add_argument('-j', default=4, type=int, metavar='N',
                    help='number of dataloading threads')
parser.add_argument('--epoch', default=30, type=int, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='start training from epoch N')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='B',
                    help='SGD mini-batch size')
parser.add_argument('-lr', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='SGD momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='use nesterov momentum')
parser.add_argument('-o', '--output-prefix', default='model-', type=str,
                    metavar='PREFIX',
                    help='pathname prefix of saved model')

def main():
    args = parser.parse_args()

    with open(args.imagelist, 'r') as fd:
        pathnames = [s.rstrip() for s in fd.readlines()]

    dataset = TrainDataset(pathnames)
    data_loader = train.utils.data.DataLoader(
        dataset,
        barth_size=args.batch_size,
        shuffle=True,
        num_workers=args.j,
        pin_memory=True
    )

    model = Sepconv().cuda()
    loss = torch.nn.L1Loss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=0.0,
        dampening=0.0,
        nesterov=args.nesterov
    )

    for epoch in range(args.start_epoch, args.epoch+1):
        train_epoch(epoch, data_loader, model, loss, optimizer)

        filename = args.output_prefix + str(epoch).rjust(3, '0')
        save_model(epoch, model, optimizer, filename)

def train_epoch(epoch, data_loader, model, loss, optimizer):
    for i, d in enumerate(data_loader):
        [input1, ground_truth, input2] = d
        input1.cuda()
        input2.cuda()
        ground_truth.cuda(async=True)

        input1_var = torch.autograd.Variable(input1)
        input2_var = torch.autograd.Variable(input2)
        target_var = touch.autograd.Variable(ground_truth)

def save_model(epoch, model, optimizer, filename):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
        },
        filename
    )

class Sepconv(torch.nn.Module):
    def __init__(self):
        super(Sepconv, self).__init__()

    def forward(self, input1, input2):
        return

class TrainDataset(torch.utils.data.Dataset):
    """Trainging dataset"""

    def __init__(self, pathnames):
        self.pathnames = pathnames

    def __len__(self):
        return len(self.pathnames)

    def __getitem__(self, idx):
        names = [self.pathnames[idx] + '_{}.png'.format(i) for i in range(3)]
        imgs = [skimage.io.imread(f) for f in names]
        """
        imput img size 150x150, output patch size 128x128
        sampled x, y: 0 <= x, y <= 22
        """

        # random shift & crop
        dx = np.random.randint(-3, 4)
        dy = np.random.randint(-3, 4)
        sx = np.random.randint(abs(dx), 22 - abs(dx) + 1)
        sy = np.random.randint(abs(dy), 22 - abs(dy) + 1)

        imgs[0] = imgs[0][sx-dx:sx-dx+128, sy-dy:sy-dy+128]
        imgs[1] = imgs[1][sx:sx+128, sy:sy+128]
        imgs[2] = imgs[2][sx+dx:sx+dx+128, sy+dy:sy+dy+128]
        # random flip
        if np.random.randint(2):
            imgs = [np.flip(img, 0) for img in imgs]
        if np.random.randint(2):
            imgs = [np.flip(img, 1) for img in imgs]
        if np.random.randint(2):
            imgs = [np.transpose(img) for img in imgs]

        # random swap first and last frame
        if np.random.randint(2):
            imgs = [imgs[2], imgs[1], imgs[0]]

        # convert to float
        imgs = [skimage.util.img_as_float32(img) for img in imgs]

        # convert to torch tensor
        # H x W x C -> C x H x W
        imgs = [torch.from_numpy(img.transpose( (2, 0, 1) )) for img in imgs]

        return imgs

if __name__ == '__main__':
    main()
