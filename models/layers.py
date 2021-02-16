import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.init_weights import init_weights


# 注意力机制
class AttentionBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_size, in_size, 3, 1, 1),
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_size, out_size, 3, 1, 1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.attention(x)
        return out


# res2net加入attention
class AR2Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, base_width=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            down_sample: None when stride = 1
            base_width: basic width of conv3x3
            scale: number of scale.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """

        super(AR2Block, self).__init__()
        down_sample = None
        if stride != 1 or inplanes != planes * self.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.attention = AttentionBlock(inplanes, planes)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        attention = self.attention(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        out = out *(1+attention)
        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


# res2next基础结构
class Res2XBlock(nn.Module):
    expansion = 1  # 扩展倍数

    def __init__(self, inplanes, planes, base_width, cardinality, stride=1, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            base_width: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            stype: 'normal': normal set. 'stage': frist blokc of a new stage.
        """
        super(Res2XBlock, self).__init__()

        down_sample = None
        if stride != 1 or inplanes != planes * self.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

        D = int(math.floor(planes * (base_width / 64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C * scale, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False))
            bns.append(nn.BatchNorm2d(D * C))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(D * C * scale, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.down_sample = down_sample
        self.width = D * C
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


# res2net的基础结构
class Res2Block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, base_width=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            base_width: basic width of conv3x3
            scale: number of scale.
            stype: 'normal': normal set. 'stage': first block of a new stage.
        """

        super(Res2Block, self).__init__()
        down_sample = None
        if stride != 1 or inplanes != planes * self.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


# encoder加入res2x
class UnetConv2Res2x(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        """
        :param in_size:输入channel
        :param out_size:输出channel
        :param is_batchnorm:是否使用batchnorm
        :param n:卷积次数
        :param ks: 卷积核长度
        :param stride:步长
        :param padding:填充
        """
        super(UnetConv2Res2x, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(Res2XBlock(in_size, out_size, base_width=26, cardinality=4),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)

                # res = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))
                #
                # setattr(self, 'res%d' % i, res)

                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(Res2XBlock(in_size, out_size, 26, 4),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = F.relu(conv(x))

        return x


# encoder加入res2
class UnetConv2Res2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        """
        :param in_size:输入channel
        :param out_size:输出channel
        :param is_batchnorm:是否使用batchnorm
        :param n:卷积次数
        :param ks: 卷积核长度
        :param stride:步长
        :param padding:填充
        """
        super(UnetConv2Res2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(Res2Block(in_size, out_size),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)

                # res = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))
                #
                # setattr(self, 'res%d' % i, res)

                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(Res2Block(in_size, out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = F.relu(conv(x))

        return x


# encoder加入res2和attention
class UnetConv2AttentionRes2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        """
        :param in_size:输入channel
        :param out_size:输出channel
        :param is_batchnorm:是否使用batchnorm
        :param n:卷积次数
        :param ks: 卷积核长度
        :param stride:步长
        :param padding:填充
        """
        super(UnetConv2AttentionRes2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(AR2Block(in_size, out_size),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)

                # res = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))
                #
                # setattr(self, 'res%d' % i, res)

                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(AR2Block(in_size, out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = F.relu(conv(x))

        return x


# encoder加入res
class UnetConv2Res(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        """
        :param in_size:输入channel
        :param out_size:输出channel
        :param is_batchnorm:是否使用batchnorm
        :param n:卷积次数
        :param ks: 卷积核长度
        :param stride:步长
        :param padding:填充
        """
        super(UnetConv2Res, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)

                res = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))

                setattr(self, 'res%d' % i, res)

                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)

                res = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p))

                setattr(self, 'res%d' % i, res)

                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            res = getattr(self, 'res%d' % i)

            x = F.relu(conv(x) + res(x))

        return x


# 原始encoder
class UnetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        """
        :param in_size:输入channel
        :param out_size:输出channel
        :param is_batchnorm:是否使用batchnorm
        :param n:卷积次数
        :param ks: 卷积核长度
        :param stride:步长
        :param padding:填充
        """
        super(UnetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(UnetUp, self).__init__()
        # self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.conv = UnetConv2(out_size * 2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)


class UnetUpOrigin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(UnetUpOrigin, self).__init__()
        # self.conv = unetConv2(out_size*2, out_size, False)
        if is_deconv:
            self.conv = UnetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = UnetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        # print(self.n_concat)
        # print(input)
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
