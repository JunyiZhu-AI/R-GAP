import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def crossentropy_for_onehot(inputs, target):
    m = nn.LogSoftmax(dim=1)
    output = torch.mean(torch.sum(-target * m(inputs), 1))
    return output


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def logistic_loss(target, inputs):
    m = nn.Sigmoid()
    pred = torch.squeeze(m(inputs), -1)
    return torch.mean(-(target*torch.log(pred)+(1-target)*torch.log(1-pred)))


def weight_init(m):
    '''
    Apply this when study the effect of adding noise. See Appendix C.
    '''
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0, 0.01)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)


class GradientMetrics:
    def __init__(self, mode):
        if mode.lower() == 'l2':
            self.fn = nn.MSELoss(reduction='sum')
        elif mode.lower() == 'cos':
            self.fn = nn.CosineSimilarity(dim=0)
        else:
            raise ValueError("GradientMetrics: Unknown mode.")
        self.mode = mode

    def __call__(self, inputs, target):
        output = 0
        if self.mode.lower() == 'l2':
            for i, t in zip(inputs, target):
                output += self.fn(i, t)
        elif self.mode.lower() == 'cos':
            gx = []
            gy = []
            for i, t in zip(inputs, target):
                gx.append(i.view(-1))
                gy.append(t.view(-1))
            output = 1 - self.fn(torch.cat(gx), torch.cat(gy))
        return output


def inverse_leakyrelu(x, slope):
    return np.array([a / slope if a < 0 else a for a in x]).astype('float32')


def derive_leakyrelu(x, slope):
    return np.array([slope if a < 0 else 1 for a in x]).reshape(1, -1).astype('float32')


def inverse_sigmoid(x):
    return np.array([-np.log(1/a - 1) for a in x]).astype('float32')


def derive_sigmoid(x):
    return np.array([a*(1-a) for a in x]).reshape(1, -1).astype('float32')


def inverse_identity(x):
    return x


def derive_identity(x):
    return np.ones(x.shape).reshape(1, -1).astype('float32')


def show_images(images, path, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure('origin')
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.gray()
        plt.imshow(image)
        plt.axis('off')
        a.set_title(title)
    plt.savefig(path)
