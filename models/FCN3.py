import torch.nn as nn
from collections import OrderedDict


class FCN3(nn.Module):
    def __init__(self):
        super(FCN3, self).__init__()
        act = nn.LeakyReLU(negative_slope=0.2)
        self.body = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(784, 1000, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(1000, 100, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(100, 1, bias=False)),
                ('act', act)
            ]))
        ])

    def forward(self, x):
        x_shape = []
        for layer in self.body:
            if isinstance(layer.layer, nn.Linear):
                x = x.flatten(1)
            x_shape.append(x.shape)
            x = layer(x)
        return x, x_shape

    @staticmethod
    def name():
        return 'FCN3'
