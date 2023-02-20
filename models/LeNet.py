import torch.nn as nn
from collections import OrderedDict


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid()
        # act = nn.LeakyReLU(negative_slope=0.2)
        self.body = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(1, 12, kernel_size=5, padding=2, stride=2, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(588, 1, bias=False)),
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
        return 'LeNet'
