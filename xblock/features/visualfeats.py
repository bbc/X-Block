#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : visualfeats.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 08.02.2021
# Last Modified Date: 09.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>

# import third-party

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__(pretrained=True, kernel_size=3)

        # third party calls
        self.vgg = third-party(pretrained=pretrained)
        self.model = third-party.Sequential(*(list(self.vgg.children())[:-1]))
        self.pooling = third-party.MaxPool2d(kernel_size=kernel_size)
        self.flat_layer = third-party.Flatten()
        # third party call ends

    def forward(self, x):
        x = self.model(x)
        x = self.pooling(x)
        out = self.flat_layer(x)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__(pretrained=True, kernel_size=3)
        self.resent = resnet18(pretrained=pretrained)
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.pooling = torch.nn.MaxPool2d(kernel_size=kernel_size)
        self.flat_layer = nn.Flatten()

    def forward(self, x):
        x = self.model(x)
        x = self.pooling(x)
        out = self.flat_layer(x)
        return out


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.model = torch.nn.Sequential(*(list(mobilenet.children())[:-1]))
        self.pooling = torch.nn.MaxPool2d(kernel_size=3)
        self.flat_layer = nn.Flatten()

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

        self.fc_layer1 = nn.Sequential(nn.Linear(7680, 6000), nn.BatchNorm1d(6000))
        self.d1 = nn.Dropout(0.6)
        self.fc_layer2 = nn.Sequential(nn.Linear(6000, 3000), nn.BatchNorm1d(3000))
        self.d2 = nn.Dropout(0.6)
        self.fc3 = nn.Linear(3000, 768)

    def forward(self, x):
        x1 = self.vgg(x)
        x2 = self.res(x)
        x3 = self.mobile(x)
        feats = torch.cat([x1, x2, x3], dim=1)
        feats = self.d1(self.fc_layer1(feats))
        feats = self.d2(self.fc_layer2(feats))
        out = self.fc3(feats)
        return out

class TransformerImageFeats(nn.Module):
    def __init__(self, num_img_embs=9):
        super(TransformerImageFeats, self).__init__()
        self.model = nn.Sequential(*list(resnet152(pretrained=True).children())[:-2])

        if num_img_embs in [1, 2, 3, 5, 7]:
            self.pool = nn.AdaptiveAvgPool2d((num_img_embs, 1))
        elif num_img_embs == 4:
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
        elif num_img_embs == 6:
            self.pool = nn.AdaptiveAvgPool2d((3, 2))
        elif num_img_embs == 8:
            self.pool = nn.AdaptiveAvgPool2d((4, 2))
        elif num_img_embs == 9:
            self.pool = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x):
        return (
            torch.flatten(self.pool(self.model(x)), start_dim=2)
            .transpose(1, 2)
            .contiguous()
        )
