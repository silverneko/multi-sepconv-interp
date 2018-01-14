import argparse
import os
import time
import numpy as np
import progressbar
import skimage.io
import skimage.util
import torch
import torch.utils.data
from Sepconv import Sepconv
from Sepconv2 import Sepconv2

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('imagelist', metavar='LIST',
                    help='list of image pathnames')
parser.add_argument('-j', default=4, type=int, metavar='N',
                    help='number of dataloading threads')
parser.add_argument('--dim', default=1, type=int, metavar='D',
                    help='model estimates filter of dimension D')
parser.add_argument('--epoch', default=30, type=int, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='start training from epoch N')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='B',
                    help='SGD mini-batch size (multiple of 4)')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='SGD momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='use nesterov momentum')
parser.add_argument('--l2', default=0.0, type=float, metavar='L2',
                    help='parameter for l2 regularization')
parser.add_argument('-o', '--output-prefix', default='model-', type=str,
                    metavar='PREFIX',
                    help='pathname prefix of saved model')
parser.add_argument('--print-freq', default=10, type=int, metavar='FREQ',
                    help='print status every FREQ minibatch')
parser.add_argument('--model', type=str, metavar='MODEL',
                    help='load model state')
parser.add_argument('--load-optimizer', action='store_true',
                    help='load optimizer state from MODEL')
parser.add_argument('--sgd', action='store_true',
                    help='use SGD')
parser.add_argument('--adamax', action='store_true',
                    help='use adamax')
parser.add_argument('--rmsprop', action='store_true',
                    help='use rmsprop')
parser.add_argument('--mse', action='store_true')
parser.add_argument('--model-type', default='v1', type=str, metavar='TYPE',
                    help='use model v1 or v2')

subbatch_size = 4

def main():
    global args
    args = parser.parse_args()

    with open(args.imagelist, 'r') as fd:
        pathnames = [s.rstrip() for s in fd.readlines()]

    dataset = TrainDataset(pathnames)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=subbatch_size,
        shuffle=True,
        num_workers=args.j,
        pin_memory=True
    )

    if args.model_type == 'v1':
        model = Sepconv(args.dim)
    elif args.model_type == 'v2':
        model = Sepconv2(args.dim, dh=64)
    model = model.cuda()

    if args.mse:
        loss = torch.nn.MSELoss().cuda()
    else:
        loss = torch.nn.L1Loss().cuda()

    if args.adamax:
        optimizer = torch.optim.Adamax(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.l2
        )
    elif args.rmsprop:
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.l2
        )
    elif args.sgd:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            dampening=0.0,
            nesterov=args.nesterov,
            weight_decay=args.l2
        )

    if args.model:
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['model'])
        if args.load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(args.start_epoch, args.start_epoch + args.epoch):
        train_epoch(epoch, data_loader, model, loss, optimizer)

        filename = args.output_prefix + str(epoch).rjust(3, '0')
        save_model(epoch, model, optimizer, filename)

def train_epoch(epoch, data_loader, model, loss, optimizer):
    subbatch_count = int(args.batch_size / subbatch_size)
    batch_count = int(np.ceil(len(data_loader) / subbatch_count))
    batch_i = 0
    train_loss = AverageMeter('Loss')
    batch_time = AverageMeter('Batch Time')
    subbatch_loss = 0
    subbatch_loss_n = 0
    subbatch_time = 0
    batch_time_end = elapse_time = time.time()

    model.train()
    for i, d in enumerate(data_loader):
        [input1, ground_truth, input2] = d
        input1 = input1.cuda()
        input2 = input2.cuda()
        ground_truth = ground_truth.cuda(async=True)

        input1_var = torch.autograd.Variable(input1)
        input2_var = torch.autograd.Variable(input2)
        target_var = torch.autograd.Variable(ground_truth)

        output_var = model(input1_var, input2_var)
        #output_var = torch.clamp(output_var, min=0.0, max=1.0)
        loss_var = loss(output_var[:,:,25:-25,25:-25], target_var[:,:,25:-25,25:-25])
        loss_var = loss_var / subbatch_count

        if i % subbatch_count == 0:
            optimizer.zero_grad()
        #hacky, bacause gradient accum over subbatches, need to normalize it
        #loss_var.backward()
        loss_var.backward()
        if (i+1) % subbatch_count == 0:
            optimizer.step()

        subbatch_loss += loss_var.data[0] * input1.size(0) * subbatch_count
        subbatch_loss_n += input1.size(0)
        subbatch_time += time.time() - batch_time_end
        batch_time_end = time.time()
        if (i+1) % subbatch_count == 0:
            batch_i += 1
            if args.mse:
                train_loss.update(np.sqrt(subbatch_loss / subbatch_loss_n))
            else:
                train_loss.update(subbatch_loss / subbatch_loss_n)
            batch_time.update(subbatch_time)
            subbatch_loss = 0
            subbatch_loss_n = 0
            subbatch_time = 0
            if batch_i % args.print_freq == 0:
                print (
                    'Epoch: {epoch} ({batch}/{batch_count}) | '
                    '{loss} | {time} | Elapse: {elapse:.2f}'
                    .format(
                        epoch=epoch,
                        batch=batch_i,
                        batch_count=batch_count,
                        loss=train_loss,
                        time=batch_time,
                        elapse=time.time() - elapse_time
                    )
                )
            if batch_i % 20 == 0:
                save_model(epoch, model, optimizer, 'checkpoint/checkpoint-{}-{}'.format(epoch, batch_i))

    # Last batch may be a partial batch, handle it elegantly
    if (i+1) % subbatch_count != 0:
        optimizer.step()

def save_model(epoch, model, optimizer, filename):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
        },
        filename
    )

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
            imgs = [np.transpose(img, [1, 0, 2]) for img in imgs]

        # random swap first and last frame
        if np.random.randint(2):
            imgs = [imgs[2], imgs[1], imgs[0]]

        # convert to float
        imgs = [skimage.util.img_as_float32(img) for img in imgs]

        # convert to torch tensor
        # H x W x C -> C x H x W
        imgs = [torch.from_numpy(img.transpose( (2, 0, 1) )) for img in imgs]

        return imgs

class AverageMeter(object):
    """Maintain a variable's current and average value (moving average)"""

    def __init__(self, name=None):
        self.reset(name)

    def reset(self, name):
        self.name = name
        self.val = 0.0
        self.avg = 0.0
        self.cum = 0.0
        self.n = 0.0
        self.alpha = 0.05

    def __str__(self):
        if self.name is None:
            return '{:.4f} ({:.4f})'.format(self.val, self.avg)
        return '{}: {:.4f} ({:.4f})'.format(self.name, self.val, self.avg)

    def update(self, val, n=1):
        self.val = val
        self.cum += val * n
        self.n += n
        if self.n == 1:
            self.avg = val
        else:
            self.avg = self.alpha * val + (1 - self.alpha) * self.avg

if __name__ == '__main__':
    main()
