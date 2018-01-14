import torch

class Sepconv2(torch.nn.Module):
    def __init__(self, dim=1, dh=32):
        super(Sepconv2, self).__init__()
        self.dim = dim
        self.dh = dh

        def EncodeLayer(inChannel, outChannel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=inChannel, out_channels=outChannel,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(outChannel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=outChannel, out_channels=outChannel,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(outChannel),
                torch.nn.ReLU(),
            )

        def PoolLayer():
            return torch.nn.AvgPool2d(kernel_size=2, stride=2)

        def DecodeLayer(inChannel, outChannel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=inChannel, out_channels=outChannel,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(outChannel),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=outChannel, out_channels=outChannel,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(outChannel),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            )

        def Subnet(inChannel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=inChannel, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=64,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=64, out_channels=51,
                                kernel_size=3, stride=1, padding=1),
                #torch.nn.BatchNorm2d(51),
                torch.nn.ReLU(),
                torch.nn.Upsample(scale_factor=2, mode='bilinear'),
                torch.nn.Conv2d(in_channels=51, out_channels=51,
                                kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        """
        self.Scale0 = torch.nn.Conv2d(in_channels=3, out_channels=3,
                                      kernel_size=3, stride=1, padding=1)
        self.DownSample = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.UpSample1 = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.UpSample2 = torch.nn.Upsample(scale_factor=4, mode='nearest')
        self.UpSample3 = torch.nn.Upsample(scale_factor=8, mode='nearest')
        self.UpSample4 = torch.nn.Upsample(scale_factor=16, mode='nearest')
        self.Conv0 = torch.nn.Conv2d(in_channels=3*5, out_channels=16,
                                     kernel_size=3, stride=1, padding=1)
        """
        self.Conv0 = torch.nn.Conv2d(in_channels=3, out_channels=16,
                                     kernel_size=3, stride=1, padding=1)

        self.Conv1 = EncodeLayer(32, 32)
        self.Pool1 = PoolLayer()

        self.Conv2 = EncodeLayer(32, 64)
        self.Pool2 = PoolLayer()

        self.Conv3 = EncodeLayer(64, 64)
        self.Pool3 = PoolLayer()

        self.Conv4 = EncodeLayer(64, 64)
        self.Pool4 = PoolLayer()

        self.Conv5 = EncodeLayer(64, 128)
        self.Pool5 = PoolLayer()

        self.Conv6 = DecodeLayer(128, 128)
        #concat6 = torch.cat([conv5, conv6], 1)

        self.Conv7 = DecodeLayer(128+128, 128)
        #concat7 = torch.cat([conv4, conv7], 1)

        self.Conv8 = DecodeLayer(64+128, 64)
        #concat8 = torch.cat([conv3, conv8], 1)

        self.Conv9 = DecodeLayer(64+64, 64)
        #concat9 = torch.cat([conv2, conv9], 1)

        subnet_dim = 64 + 64
        self.Vert1 = torch.nn.ModuleList([Subnet(subnet_dim) for i in range(dim)])
        self.Hori1 = torch.nn.ModuleList([Subnet(subnet_dim) for i in range(dim)])
        self.Vert2 = torch.nn.ModuleList([Subnet(subnet_dim) for i in range(dim)])
        self.Hori2 = torch.nn.ModuleList([Subnet(subnet_dim) for i in range(dim)])

        pad_width = int((51 - 1) / 2)
        self.PadInput = torch.nn.ReplicationPad2d([pad_width] * 4)

        #self.ConvOut = torch.nn.Conv2d(in_channels=3, out_channels=3,
        #                               kernel_size=3, stride=1, padding=1)
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
        """ Multiscale input
        input1_scale0 = self.Scale0(input1)
        input1_scale1 = self.DownSample(input1_scale0)
        input1_scale2 = self.DownSample(input1_scale1)
        input1_scale3 = self.DownSample(input1_scale2)
        input1_scale4 = self.DownSample(input1_scale3)
        input1_resize1 = self.UpSample1(input1_scale1)
        input1_resize2 = self.UpSample2(input1_scale2)
        input1_resize3 = self.UpSample3(input1_scale3)
        input1_resize4 = self.UpSample4(input1_scale4)

        input1_concat = torch.cat([input1_scale0, input1_resize1, input1_resize2, input1_resize3, input1_resize4], 1)

        input2_scale0 = self.Scale0(input2)
        input2_scale1 = self.DownSample(input2_scale0)
        input2_scale2 = self.DownSample(input2_scale1)
        input2_scale3 = self.DownSample(input2_scale2)
        input2_scale4 = self.DownSample(input2_scale3)
        input2_resize1 = self.UpSample1(input2_scale1)
        input2_resize2 = self.UpSample2(input2_scale2)
        input2_resize3 = self.UpSample3(input2_scale3)
        input2_resize4 = self.UpSample4(input2_scale4)

        input2_concat = torch.cat([input2_scale0, input2_resize1, input2_resize2, input2_resize3, input2_resize4], 1)

        input1_conv = self.Conv0(input1_concat)
        input2_conv = self.Conv0(input2_concat)
        """
        input1_conv = self.Conv0(input1)
        input2_conv = self.Conv0(input2)
        Input = torch.cat([input1_conv, input2_conv], 1)

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
        concat6 = torch.cat([conv5, conv6], 1)

        conv7 = self.Conv7(concat6)
        concat7 = torch.cat([conv4, conv7], 1)

        conv8 = self.Conv8(concat7)
        concat8 = torch.cat([conv3, conv8], 1)

        conv9 = self.Conv9(concat8)
        concat9 = torch.cat([conv2, conv9], 1)

        [batch, channel, height, width] = list(input1.size())

        k1v = torch.stack([self.Vert1[i](concat9) for i in range(self.dim)], dim=-1)
        k1h = torch.stack([self.Hori1[i](concat9) for i in range(self.dim)], dim=-1)
        k2v = torch.stack([self.Vert2[i](concat9) for i in range(self.dim)], dim=-1)
        k2h = torch.stack([self.Hori2[i](concat9) for i in range(self.dim)], dim=-1)

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
        #Output = self.ConvOut(Output)
        #Output = self.ClipOutput(Output)

        return Output
