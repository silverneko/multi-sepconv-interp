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
    description='test.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--model', type=str, metavar='MODEL',
                    help='model parameter')
parser.add_argument('--test', action='store_true')
parser.add_argument('--first', type=str, metavar='FIRST')
parser.add_argument('--second', type=str, metavar='SECOND')
parser.add_argument('-o', '--output', default='out.png', type=str,
                    metavar='OUTPUT',
                    help='output file name')

def main():
    global args
    args = parser.parse_args()

    model = Sepconv()
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['model'])
    model = model

    img0 = skimage.io.imread(args.first)
    img1 = skimage.io.imread(args.second)

    img0 = skimage.util.img_as_float32(img0)
    img1 = skimage.util.img_as_float32(img1)
    h, w, c = img0.shape
    newh = int(np.ceil(h / 32)) * 32
    neww = int(np.ceil(w / 32)) * 32

    img0 = np.pad(img0, ((0, newh-h), (0, neww-w), (0,0)), 'edge')
    img1 = np.pad(img1, ((0, newh-h), (0, neww-w), (0,0)), 'edge')

    img0 = np.ascontiguousarray(img0)
    img1 = np.ascontiguousarray(img1)

    img0 = torch.from_numpy(img0.transpose( (2, 0, 1) ))
    img1 = torch.from_numpy(img1.transpose( (2, 0, 1) ))

    img0 = img0.unsqueeze(0)
    img1 = img1.unsqueeze(0)

    res = test(model, img0, img1)

    res = res[0].data.numpy()
    res = res.transpose((1, 2, 0))
    res = res[:h, :w]
    skimage.io.imsave(args.output, res)

def test(model, img0, img1):
    model.eval()

    img0 = torch.autograd.Variable(img0, volatile=True)
    img1 = torch.autograd.Variable(img1, volatile=True)
    res = model(img0, img1)
    return res

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

if __name__ == '__main__':
    main()
