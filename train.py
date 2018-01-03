import argparse
import os
import time
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
                    help='SGD mini-batch size (multiple of 4)')
parser.add_argument('-lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='SGD momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='use nesterov momentum')
parser.add_argument('-o', '--output-prefix', default='model-', type=str,
                    metavar='PREFIX',
                    help='pathname prefix of saved model')
parser.add_argument('--print-freq', default=10, type=int, metavar='FREQ',
                    help='print status every FREQ minibatch')
parser.add_argument('--model', type=str, metavar='MODEL',
                    help='load model state')
parser.add_argument('--adamax', action='store_true',
                    help='use adamax instead of SGD')

def main():
    global args
    args = parser.parse_args()

    with open(args.imagelist, 'r') as fd:
        pathnames = [s.rstrip() for s in fd.readlines()]

    dataset = TrainDataset(pathnames)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=args.j,
        pin_memory=True
    )

    model = Sepconv()
    if args.model:
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    loss = torch.nn.L1Loss().cuda()

    if args.adamax:
        optimizer = torch.optim.Adamax(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-6
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=1e-6,
            dampening=0.0,
            nesterov=args.nesterov
        )

    for epoch in range(args.start_epoch, args.epoch+1):
        train_epoch(epoch, data_loader, model, loss, optimizer)

        filename = args.output_prefix + str(epoch).rjust(3, '0')
        save_model(epoch, model, optimizer, filename)

def train_epoch(epoch, data_loader, model, loss, optimizer):
    subbatch_count = (args.batch_size / 4)
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
        loss_var = loss(output_var, target_var)

        if i % subbatch_count == 0:
            optimizer.zero_grad()
        #hacky, bacause gradient accum over subbatches, need to normalize it
        #loss_var.backward()
        loss_var.backward(torch.cuda.FloatTensor([1 / subbatch_count]))
        if (i+1) % subbatch_count == 0:
            optimizer.step()


        subbatch_loss += loss_var.data[0]
        subbatch_loss_n += input1.size(0)
        subbatch_time += time.time() - batch_time_end
        batch_time_end = time.time()
        if (i+1) % subbatch_count == 0:
            batch_i += 1
            train_loss.update(subbatch_loss / subbatch_loss_n)
            batch_time.update(subbatch_time)
            subbatch_loss = 0
            subbatch_loss_n = 0
            subbatch_time = 0
            if batch_i % args.print_freq == 0:
                print (
                    'Epoch: {epoch}({batch}/{batch_count}) | '
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

class Sepconv(torch.nn.Module):
    def __init__(self):
        super(Sepconv, self).__init__()

        def TripleConv2d(inChannel, outChannel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=inChannel, out_channels=outChannel,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=outChannel, out_channels=outChannel,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=outChannel, out_channels=outChannel,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU()
            )

        def UpsampleLayer(inChannel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(in_channels=inChannel, out_channels=inChannel,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU()
            )

        def Subnet():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=51,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(in_channels=51, out_channels=51,
                                kernel_size=3, stride=1, padding=1),
            )

        self.Conv1 = TripleConv2d(6, 32)
        self.Pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.Conv2 = TripleConv2d(32, 64)
        self.Pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.Conv3 = TripleConv2d(64, 128)
        self.Pool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.Conv4 = TripleConv2d(128, 256)
        self.Pool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.Conv5 = TripleConv2d(256, 512)
        self.Pool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.Conv6 = TripleConv2d(512, 512)
        self.Upsample6 = UpsampleLayer(512)

        self.Conv7 = TripleConv2d(512, 256)
        self.Upsample7 = UpsampleLayer(256)

        self.Conv8 = TripleConv2d(256, 128)
        self.Upsample8 = UpsampleLayer(128)

        self.Conv9 = TripleConv2d(128, 64)
        self.Upsample9 = UpsampleLayer(64)

        self.Vertical1 = Subnet()
        self.Horizontal1 = Subnet()
        self.Vertical2 = Subnet()
        self.Horizontal2 = Subnet()

        pad_width = int((51 - 1) / 2)
        self.PadInput = torch.nn.ReplicationPad2d([pad_width] * 4)

    def forward(self, input1, input2):
        Input = torch.cat([input1, input2], 1)

        conv1 = self.Conv1(Input)
        pool1 = self.Pool1(conv1)

        conv2 = self.Conv2(pool1)
        pool2 = self.Pool2(conv2)

        conv3 = self.Conv3(pool2)
        pool3 = self.Pool3(conv3)

        conv4 = self.Conv4(pool3)
        pool4 = self.Pool4(conv4)

        conv5 = self.Conv5(pool4)
        pool5 = self.Pool5(conv5)

        conv6 = self.Conv6(pool5)
        up6   = self.Upsample6(conv6)
        comb6 = up6 + conv5

        conv7 = self.Conv7(comb6)
        up7   = self.Upsample7(conv7)
        comb7 = up7 + conv4

        conv8 = self.Conv8(comb7)
        up8   = self.Upsample8(conv8)
        comb8 = up8 + conv3

        conv9 = self.Conv9(comb8)
        up9   = self.Upsample9(conv9)
        comb9 = up9 + conv2

        k1v = self.Vertical1(comb9)
        k1h = self.Horizontal1(comb9)
        k2v = self.Vertical2(comb9)
        k2h = self.Horizontal2(comb9)

        padded1 = self.PadInput(input1)
        padded2 = self.PadInput(input2)

        Output = torch.zeros_like(input1)
        [batch, channel, height, width] = list(Output.size())


        for h in range(height):
            P1 = padded1[:,:,h:h+51,:].unfold(3, 51, 1).permute(0, 1, 3, 2, 4)
            P2 = padded2[:,:,h:h+51,:].unfold(3, 51, 1).permute(0, 1, 3, 2, 4)

            K1v = k1v[:, :, h, :].permute(0, 2, 1).unsqueeze(1).unsqueeze(3)
            K1h = k1h[:, :, h, :].permute(0, 2, 1).unsqueeze(1).unsqueeze(4)
            K2v = k2v[:, :, h, :].permute(0, 2, 1).unsqueeze(1).unsqueeze(3)
            K2h = k2h[:, :, h, :].permute(0, 2, 1).unsqueeze(1).unsqueeze(4)

            I = (torch.matmul(K1v, torch.matmul(P1, K1h)) +
                torch.matmul(K2v, torch.matmul(P2, K2h))).squeeze(4).squeeze(3)
            Output[:,:,h] = torch.clamp(I, min=0.0, max=1.0)
        return Output

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
    """Maintain a variable's current and average value"""

    def __init__(self, name=None):
        self.reset(name)

    def reset(self, name):
        self.name = name
        self.val = 0.0
        self.avg = 0.0
        self.cum = 0.0
        self.n = 0.0

    def __str__(self):
        if self.name is None:
            return '{:.4f} ({:.4f})'.format(self.val, self.avg)
        return '{}: {:.4f} ({:.4f})'.format(self.name, self.val, self.avg)

    def update(self, val, n=1):
        self.val = val
        self.cum += val * n
        self.n += n
        if self.n != 0:
            self.avg = self.cum / self.n

if __name__ == '__main__':
    main()
