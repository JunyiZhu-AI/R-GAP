from utils import *
from conv2circulant import *

setup = {'device': 'cpu', 'dtype': torch.float32}


def logistic_loss(y, pred):
    y = torch.tensor(y).to(**setup)
    pred = torch.squeeze(pred, -1)
    return torch.mean(-(y*torch.log(pred)+(1-y)*torch.log(1-pred)))


def inverse_udldu(udldu):
    '''derive u from udldu using gradient descend based method'''
    lr = 0.01
    u = torch.tensor(0).to(**setup).requires_grad_(True)
    udldu = torch.tensor(udldu).to(**setup)
    optimizer = torch.optim.Adam([u], lr=lr)
    loss_fn = nn.MSELoss()
    for i in range(30000):
        optimizer.zero_grad()
        udldu_ = -u / (1 + torch.exp(u))
        l = loss_fn(udldu_, udldu)
        l.backward()
        optimizer.step()
    udldu_ = -u / (1 + torch.exp(u))
    print(f"The error term of inversing udldu: {udldu.item()-udldu_.item():.1e}")
    return u.detach().numpy()


def peeling(in_shape, padding):
    if padding == 0:
        return np.ones(shape=in_shape, dtype=bool).squeeze()
    h, w = np.array(in_shape[-2:]) + 2*padding
    toremain = np.ones(h*w*in_shape[1], dtype=np.bool)
    if padding:
        for c in range(in_shape[1]):
            for row in range(h):
                for col in range(w):
                    if col < padding or w-col <= padding or row < padding or h-row <= padding:
                        i = c*h*w + row*w + col
                        assert toremain[i]
                        toremain[i] = False
    return toremain


def padding_constraints(in_shape, padding):
    toremain = peeling(in_shape, padding)
    P = []
    for i in range(toremain.size):
        if not toremain[i]:
            P_row = np.zeros(toremain.size, dtype=np.float32)
            P_row[i] = 1
            P.append(P_row)
    return np.array(P)


def cnn_reconstruction(in_shape, k, g, out, kernel, stride, padding):
    coors, x_len, y_len = generate_coordinates(x_shape=in_shape, kernel=kernel, stride=stride, padding=padding)
    K = aggregate_g(k=k, x_len=x_len, coors=coors)
    W = circulant_w(x_len=x_len, kernel=kernel, coors=coors, y_len=y_len)
    P = padding_constraints(in_shape=in_shape, padding=padding)
    p = np.zeros(shape=P.shape[0], dtype=np.float32)
    if np.any(P):
        a = np.concatenate((K, W, P), axis=0)
        b = np.concatenate((g.reshape(-1), out, p), axis=0)
    else:
        a = np.concatenate((K, W), axis=0)
        b = np.concatenate((g.reshape(-1), out), axis=0)
    result = np.linalg.lstsq(a, b, rcond=None)
    print(f'lstsq residual: {result[1]}, rank: {result[2]} -> {W.shape[-1]}, '
          f'max/min singular value: {result[3].max():.2e}/{result[3].min():.2e}')
    x = result[0]
    return x[peeling(in_shape=in_shape, padding=padding)], W


def fcn_reconstruction(k, gradient):
    x = [g / c for g, c in zip(gradient, k) if c != 0]
    x = np.mean(x, 0)
    return x


def r_gap(out, k, g, x_shape, weight, module):
    # obtain information of convolution kernel
    if isinstance(module.layer, nn.Conv2d):
        padding = module.layer.padding[0]
        stride = module.layer.stride[0]
    else:
        padding = 0
        stride = 1

    x, weight = cnn_reconstruction(in_shape=x_shape, k=k, g=g, kernel=weight, out=out, stride=stride, padding=padding)
    return x, weight







