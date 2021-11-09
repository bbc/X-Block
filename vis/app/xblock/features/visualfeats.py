#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : visualfeats.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# base library for extrcting visual representations

# import third_party libraries

class third_party(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = third_party(pretrained=True)
        self.model = third_partySequential(*(list(self.vgg.children())[:-1]))
        self.pooling = third_party.MaxPool2d(kernel_size=3)
        self.flat_layer = third_party.Flatten()

    def forward(self, x):
        x = self.model(x)
        x = self.pooling(x)
        out = self.flat_layer(x)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.resent = third_party(pretrained=True)
        self.model = third_party.Sequential(*(list(resnet.children())[:-1]))
        self.pooling = third_party.MaxPool2d(kernel_size=3)
        self.flat_layer = third_party.Flatten()

    def forward(self, x):
        x = self.model(x)
        #    x = self.pooling(x)
        out = self.flat_layer(x)
        return out


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.mobilenet = third_party(pretrained=True)
        self.model = third_party.Sequential(*(list(mobilenet.children())[:-1]))
        self.pooling = third_party.MaxPool2d(kernel_size=3)
        self.flat_layer = third_party.Flatten()

    def forward(self, x):
        x = self.model(x)
        x = self.pooling(x)
        out = self.flat_layer(x)
        return out


class EnsembleFeats(nn.Module):
    def __init__(self):
        super(EnsembleFeats, self).__init__()
        self.vgg = VGG()
        self.res = ResNet()
        self.mobile = MobileNet()

        self.fc_layer1 = third_party.Sequential(third_party.Linear(7680, 6000), third_party.BatchNorm1d(6000))
        self.d1 = third_party.Dropout(0.6)
        self.fc_layer2 = third_party.Sequential(third_party.Linear(6000, 3000), third_party.BatchNorm1d(3000))
        self.d2 = third_party.Dropout(0.6)
        self.fc3 = third_party.Linear(3000, 768)

    def forward(self, x):
        x1 = self.vgg(x)
        x2 = self.res(x)
        x3 = self.mobile(x)
        feats = torch.cat([x1, x2, x3], dim=1)
        feats = self.d1(self.fc_layer1(feats))
        feats = self.d2(self.fc_layer2(feats))
        out = self.fc3(feats)
        return out
