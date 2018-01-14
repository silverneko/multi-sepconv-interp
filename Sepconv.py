import torch

class Sepconv(torch.nn.Module):
    def __init__(self, dim=1, dh=32):
        super(Sepconv, self).__init__()
        self.dim = dim
        self.dh = dh

        def TripleConv2d(inChannel, outChannel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=inChannel, out_channels=outChannel,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(outChannel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=outChannel, out_channels=outChannel,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(outChannel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=outChannel, out_channels=outChannel,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(outChannel),
                torch.nn.ReLU()
            )

        def UpsampleLayer(inChannel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(in_channels=inChannel, out_channels=inChannel,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(inChannel),
                torch.nn.ReLU()
            )

        def Subnet_front():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(64),
                torch.nn.ReLU()
            )

        def Subnet_back():
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=51,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(51),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(in_channels=51, out_channels=51,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
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

        self.Vertical1 = Subnet_front()
        self.Horizontal1 = Subnet_front()
        self.Vertical2 = Subnet_front()
        self.Horizontal2 = Subnet_front()

        self.kv1s = torch.nn.ModuleList([Subnet_back() for i in range(dim)])
        self.kh1s = torch.nn.ModuleList([Subnet_back() for i in range(dim)])
        self.kv2s = torch.nn.ModuleList([Subnet_back() for i in range(dim)])
        self.kh2s = torch.nn.ModuleList([Subnet_back() for i in range(dim)])

        pad_width = int((51 - 1) / 2)
        self.PadInput = torch.nn.ReplicationPad2d([pad_width] * 4)
        self.ClipOutput = torch.nn.Sigmoid()

        self.reset_weight()

    def reset_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform(m.weight.data)
                # torch.nn.init.kaiming_uniform(m.weight.data)
                # torch.nn.init.orthogonal(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant(m.bias.data, 0.0)

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

        vert1 = self.Vertical1(comb9)
        horz1 = self.Horizontal1(comb9)
        vert2 = self.Vertical2(comb9)
        horz2 = self.Horizontal2(comb9)

        [batch, channel, height, width] = list(input1.size())

        k1v = torch.stack([self.kv1s[i](vert1) for i in range(self.dim)], dim=-1)
        k1h = torch.stack([self.kh1s[i](horz1) for i in range(self.dim)], dim=-1)
        k2v = torch.stack([self.kv2s[i](vert2) for i in range(self.dim)], dim=-1)
        k2h = torch.stack([self.kh2s[i](horz2) for i in range(self.dim)], dim=-1)

        padded1 = self.PadInput(input1)
        padded2 = self.PadInput(input2)

        Output = torch.zeros_like(input1)

        dh = self.dh
        for h in range(0, height, dh):
            P1 = [padded1[:,:,hi:hi+51].unfold(3, 51, 1) for hi in range(h, h+dh)]
            P2 = [padded2[:,:,hi:hi+51].unfold(3, 51, 1) for hi in range(h, h+dh)]
            P1 = torch.stack(P1).permute(1, 2, 0, 4, 3, 5)
            P2 = torch.stack(P2).permute(1, 2, 0, 4, 3, 5)

            K1v = k1v[:,:,h:h+dh].permute(0, 2, 3, 1, 4).unsqueeze(1)
            K1h = k1h[:,:,h:h+dh].permute(0, 2, 3, 1, 4).unsqueeze(1)
            K2v = k2v[:,:,h:h+dh].permute(0, 2, 3, 1, 4).unsqueeze(1)
            K2h = k2h[:,:,h:h+dh].permute(0, 2, 3, 1, 4).unsqueeze(1)

            I1 = (K1v * torch.matmul(P1, K1h)).sum(5).sum(4)
            I2 = (K2v * torch.matmul(P2, K2h)).sum(5).sum(4)
            Output[:,:,h:h+dh] = I1+I2
        #Output = self.ClipOutput(Output)

        return Output
