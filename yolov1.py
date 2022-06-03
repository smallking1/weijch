import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class Convention(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, conv_stride, padding):
        super(Convention, self).__init__()
        self.Conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, conv_stride, padding)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=1e-1,inplace=True)

    def forward(self,x):
        x = self.Conv1(x)
        x = self.LeakyReLU(x)
        return x

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)



class yolov1(nn.Module):
    def __init__(self, B=2, class_num=20):
        super(yolov1, self).__init__()
        self.B = B
        self.class_num = class_num

        self.Conv1 = nn.Sequential(Convention(3, 64, 7, 2, 3),
                                   nn.MaxPool2d(2, 2))

        self.Conv2 = nn.Sequential(Convention(64, 192, 3, 1, 1),
                                   nn.MaxPool2d(2, 2))

        self.Conv3 = nn.Sequential(Convention(192, 128, 1, 1, 0),
                                   Convention(128, 256, 3, 1, 1),
                                   Convention(256, 256, 1, 1, 0),
                                   Convention(256, 512, 3, 1, 1),
                                   nn.MaxPool2d(2, 2)
                                   )

        self.Conv4 = nn.Sequential(Convention(512, 256, 1, 1, 0),
                                   Convention(256, 512, 3, 1, 1),
                                   Convention(512, 256, 1, 1, 0),
                                   Convention(256, 512, 3, 1, 1),
                                   Convention(512, 256, 1, 1, 0),
                                   Convention(256, 512, 3, 1, 1),
                                   Convention(512, 256, 1, 1, 0),
                                   Convention(256, 512, 3, 1, 1),
                                   Convention(512, 512, 1, 1, 0),
                                   Convention(512, 1024, 3, 1, 1),
                                   nn.MaxPool2d(2, 2)
                                   )

        self.Conv5 = nn.Sequential(Convention(1024, 512, 1, 1, 0),
                                   Convention(512, 1024, 3, 1, 1),
                                   Convention(1024, 512, 1, 1, 0),
                                   Convention(512, 1024, 3, 1, 1),
                                   Convention(1024, 1024, 3, 1, 1),
                                   Convention(1024, 1024, 3, 2, 1)
                                   )

        self.Conv6 = nn.Sequential(Convention(1024, 1024, 3, 1, 1),
                                   Convention(1024, 1024, 3, 1, 1)
                                   )

        self.Fc7 = nn.Sequential(nn.Linear(7*7*1024, 4096),
                                   nn.LeakyReLU(negative_slope=1e-1, inplace=True),
                                   nn.Linear(4096, 7 * 7 * (self.B * 5 + self.class_num)),
                                   nn.Sigmoid()
                                   )

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.Conv6(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.Fc7(x)
        x = x.view(-1,7,7,(self.B*5 + self.class_num))
        return x

    def yolo_weight_init(self):
        for m in self.modules():
            if isinstance(m, Convention):
                m.weight_init()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()




class detnet_bottleneck(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type='A'):
        super(detnet_bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=2, bias=False,dilation=2)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes or block_type=='B':
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.layer5 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_detnet_layer(in_channels=2048)
        # self.avgpool = nn.AvgPool2d(14) #fit 448 input size
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv_end = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(128)
        self.Fc = nn.Sequential(nn.Linear(14 * 14 * 128, 4096),
                                 nn.LeakyReLU(negative_slope=1e-1, inplace=True),
                                 nn.Linear(4096, 7 * 7 * 30),
                                 nn.Sigmoid()
                                 )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_detnet_layer(self, in_channels):
        layers = []
        layers.append(detnet_bottleneck(in_planes=in_channels, planes=256, block_type='B'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        layers.append(detnet_bottleneck(in_planes=256, planes=256, block_type='A'))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        # x = x.view(-1,7,7,30)
        x = x.permute(0, 2, 3, 1)  # (-1,14,14,128)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.Fc(x)
        x = x.view(-1, 7, 7, 30)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def test():
    import torch
    from torch.autograd import Variable
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    print(model)
    #print(model)
    img = torch.rand(2,3,448,448)
    img = Variable(img)
    output = model(img)
    print(output.size())



if __name__ == '__main__':
    test()

    # output = model(data)
    # print(output.shape)





