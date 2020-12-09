import numpy as np

import torch
import torch.nn as nn

#using author created classification network for TransferLearning

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, nChannels=256, num_classes=200, pool_size=6, pool_type='max'):
        super(Classifier, self).__init__()
        nChannelsAll = nChannels * pool_size * pool_size

        layers = []
        if pool_type == 'max':
            layers.append(nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            #layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        elif pool_type == 'avg':
            layer.append(nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        layers.append(nn.BatchNorm2d(nChannels, affine=False))
        layers.append(Flatten())
        layers.append(nn.Linear(nChannelsAll, num_classes))
        self.classifier = nn.Sequential(*layers)
        self.initilize()
    
    def forward(self, feat):
        return self.classifier(feat)
    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fin = m.in_features
                fout = m.out_features
                std_val = np.sqrt(2.0/fout)
                m.weight.data.normal_(0.0, std_val)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)