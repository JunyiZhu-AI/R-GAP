import torchvision
import argparse
import os
import yaml
from utils import *
from models import CNN6, CNN6d, FCN3
from recursive_attack import r_gap, peeling, fcn_reconstruction, inverse_udldu
import matplotlib.pyplot as plt

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
parser = argparse.ArgumentParser(description="Model related arguments. For other configurations please check CONFIG file.")
parser.add_argument("-d", "--dataset", help="Choose the data source.", choices=["CIFAR10", "MNIST"], default="CIFAR10")
parser.add_argument("-i", "--index", help="Choose a specific image to reconstruct.", type=int, default=-1)
parser.add_argument("-b", "--batchsize", default=1, help="Mini-batch size", type=int)
parser.add_argument("-p", "--parameters", help="Load pre-trained model.", default=None)
parser.add_argument("-m", "--model", help="Network architecture.", choices=["CNN6", "CNN6-d", "FCN3"], default='CNN6')
args = parser.parse_args()
setup = {'device': 'cpu', 'dtype': torch.float32}
print(f'Running on {setup["device"]}, PyTorch version {torch.__version__}')


def main():
    train_sample, test_sample = dataloader(dataset=args.dataset, mode="attack", index=args.index,
                                           batchsize=args.batchsize, config=config)
    # set up inference framework
    torch.manual_seed(0)
    np.random.seed(0)
    if args.model == 'CNN6':
        net = CNN6().to(**setup).eval()
    elif args.model == 'CNN6-d':
        net = CNN6d().to(**setup).eval()
    else:
        net = FCN3().to(**setup).eval()
    pred_loss_fn = logistic_loss

    tt = torchvision.transforms.ToTensor()
    tp = torchvision.transforms.ToPILImage()
    if args.batchsize == 1:
        image, label = train_sample
        x = tt(image).unsqueeze(0).to(**setup)
    else:
        image, label = list(zip(*train_sample))
        x = [tt(im) for im in image]
        x = torch.stack(x).to(**setup)

    # load parameters
    if args.parameters:
        checkpoint = torch.load(args.parameters)
        ep = checkpoint['epoch']
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f'Load model trained with {ep} epochs.')

    # generate gradients of real data
    pred, x_shape = net(x)
    # reversed label to make sure mu is unique, just for better demonstration
    y = torch.tensor([0 if p > 0 else 1 for p in pred]).to(**setup)
    print(f'pred: {pred.detach().numpy()}, y: {y}')
    pred_loss = pred_loss_fn(inputs=pred, target=y)
    dy_dx = torch.autograd.grad(pred_loss, list(net.parameters()))
    original_dy_dx = [g.detach().clone() for g in dy_dx]

    # reconstruction procedure
    original_dy_dx.reverse()
    modules = net.body[-1::-1]
    x_shape.reverse()
    k = None
    last_weight = []

    print('****************')
    print('perform R-GAP')
    print('****************')
    for i in range(len(modules)):
        g = original_dy_dx[i].numpy()
        w = list(modules[i].layer.parameters())[0].detach().cpu().numpy()
        if k is None:
            udldu = np.dot(g.reshape(-1), w.reshape(-1))
            u = inverse_udldu(udldu)

            # For simplicity assume y as known here. For details please refer to the paper.
            y = np.array([-1 if n == 0 else n for n in y], dtype=np.float32).reshape(-1, 1)
            y = y.mean() if y.mean() != 0 else 0.1

            print(f'pred_: {u/y:.1e}, udldu: {udldu:.1e}, udldu_:{-u/(1+np.exp(u)):.1e}')
            k = -y/(1+np.exp(u))
            k = k.reshape(1, -1).astype(np.float32)

        else:
            # derive activation function
            if isinstance(modules[i].act, nn.LeakyReLU):
                da = derive_leakyrelu(x_, slope=modules[i].act.negative_slope)
            elif isinstance(modules[i].act, nn.Identity):
                da = derive_identity(x_)
            else:
                ValueError(f'Please implement the derivative function of {modules[i].act}')

            # back out neuron output
            if isinstance(modules[i].act, nn.LeakyReLU):
                out = inverse_leakyrelu(x_, slope=modules[i].act.negative_slope)
            elif isinstance(modules[i].act, nn.Identity):
                out = inverse_identity(x_)
            else:
                ValueError(f'Please implement the inverse function of {modules[i].act}')
            if hasattr(modules[i-1].layer, 'padding'):
                padding = modules[i-1].layer.padding[0]
            else:
                padding = 0

            # For a mini-batch setting, reconstruct the combination
            in_shape = np.array(x_shape[i-1])
            in_shape[0] = 1
            # peel off padded entries
            x_mask = peeling(in_shape=in_shape, padding=padding)
            k = np.multiply(np.matmul(last_weight.transpose(), k)[x_mask], da.transpose())

        if isinstance(modules[i].layer, nn.Conv2d):
            x_, last_weight = r_gap(out=out, k=k, x_shape=x_shape[i], module=modules[i], g=g, weight=w)
        else:
            # In consideration of computational efficiency, for FCN only takes gradient constraints into account.
            x_, last_weight = fcn_reconstruction(k=k, gradient=g), w

    # visualization
    x_ = x_.reshape(x.shape[-3:]).squeeze()
    if args.batchsize > 1:
        show_images(image, path=os.path.join(config['path_to_demo'], 'origin.png'), cols=len(image)//2+1)
    else:
        plt.figure('origin')
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(os.path.join(config['path_to_demo'], 'origin.png'))

    plt.figure('reconstructed')
    plt.imshow(tp(torch.tensor(x_)))
    plt.axis('off')
    plt.savefig(os.path.join(config['path_to_demo'], 'reconstructed.png'))
    plt.figure('rescale reconstructed')
    plt.imshow(tp(torch.tensor((x_-x_.min())/x_.max())))
    plt.axis('off')
    plt.savefig(os.path.join(config['path_to_demo'], 'rescale_reconstructed.png'))


if __name__ == "__main__":
    main()
