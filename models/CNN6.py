import torch.nn as nn
from collections import OrderedDict


class CNN6(nn.Module):
    def __init__(self):
        super(CNN6, self).__init__()
        act = nn.LeakyReLU(negative_slope=0.2)
        self.body = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(3, 12, kernel_size=4, padding=2, stride=2, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(12, 36, kernel_size=3, padding=1, stride=2, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(36, 36, kernel_size=3, padding=1, stride=1, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(36, 36, kernel_size=3, padding=1, stride=1, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(36, 64, kernel_size=3, padding=1, stride=2, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(3200, 1, bias=False)),
                ('act', nn.Identity())
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
        return 'CNN6'
