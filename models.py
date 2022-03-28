import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
import random
import time

class Lenet(nn.Module):

    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        clean = F.softmax(out, 1)

        return clean


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes,mode='cifar10'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if mode == 'mnist':
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, revision=True):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.layer4(out)  # 1*512*4*4

        out = self.avgpool(out)  # 1*512*1*1


        out = out.view(out.size(0), -1)

        out = self.linear(out)
        #print('out-1:',out)
        clean = F.softmax(out, 1)
        #print('softmax:',clean)
        #print('\n')

        return clean


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes,mode='mnist')


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def ResNet18_pre(num_classes):
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes,mode='mnist')


def ResNet34_pre(num_classes):
    return ResNet(PreActBlock, [3, 4, 6, 3], num_classes)


def ResNet50_pre(num_classes):
    return ResNet(PreActBlock, [3, 4, 6, 3], num_classes)


def ResNet101_pre(num_classes):
    return ResNet(PreActBlock, [3, 4, 23, 3], num_classes)


def ResNet152_pre(num_classes):
    return ResNet(PreActBlock, [3, 8, 36, 3], num_classes)

class sig_t(nn.Module):
    # tensor T to parameter
    def __init__(self, device, num_classes, init=4.5):
        super(sig_t, self).__init__()
        self.device = device
        # self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(num_classes, num_classes)))
        self.num_classes = num_classes


        self.fc = nn.Linear(num_classes, num_classes * num_classes, bias=False)

        self.ones =  torch.ones(num_classes)
        self.zeros = torch.zeros([num_classes, num_classes])
        self.w = torch.Tensor([])
        for i in range(num_classes):
            '''k = random.randint(0,self.num_classes-1)
            while k==i:
                k = random.randint(0,self.num_classes-1)'''

            temp = self.zeros.clone()
            #ind1 = temp[k].add_(1e-4)
            ind = temp[i].add_(self.ones-0.1)
            #print(temp)
            temp = temp+0.1/self.num_classes
            self.w = torch.cat([self.w, temp.detach()], 0)
            #self.w[i] = self.w[i] + 0.01
        #nn.init.kaiming_normal_(self.fc.weight)


        #print(-init)
        #self.fc.weight.data = -init * torch.ones([num_classes*num_classes,num_classes])
        #print(self.fc.weight.data[0])

        self.fc.weight.data = self.w
        #print(self.fc.weight.data[0:10][:, 0:9])
        #exit(-1)




    def forward(self, x):
        self.identity = torch.cat([torch.eye(self.num_classes).unsqueeze(0) for i in range(x.size(0))], 0)
        self.identity = self.identity.to(self.device)
        co = torch.ones(self.num_classes, self.num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        co = torch.cat([co.unsqueeze(0) for i in range(x.size(0))], 0)
        self.co = co.to(self.device)


        out = x.view(x.size(0), -1)
        out = self.fc(out)
        out = out.view(x.size(0), self.num_classes, -1)
        #out = torch.sigmoid(out)
        out = torch.clamp(out, min=1e-5,max=1-1e-5)
        #print('Qian:')
        #print(out[0])
        #print(self.fc.weight.data[0:10][:, 0:9])
        #out = self.identity.detach() + out*self.co.detach()
        out = F.normalize(out, p=1, dim=2).to(self.device)
        #print('Hou:')
        #print(out[0])
        return out
