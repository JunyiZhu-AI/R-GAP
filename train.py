import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
import os
import datetime
from utils import dataloader, CLASSES, logistic_loss
from models import CNN6, CNN6d

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
parser = argparse.ArgumentParser(description="Please input model related arguments here. For more meta arguments please check CONFIG file.")
parser.add_argument("-d", "--dataset", help="Choose the data source.", choices=["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST"], default="FashionMNIST")
parser.add_argument("-b", "--batchsize", default=5, help="Mini-batch size", type=int)
parser.add_argument("-a", "--augmentation", type=bool, default=False, help="Data augmentation")
parser.add_argument("-e", "--epoch", help="Training epoches.", default=5, type=int)
parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float)
parser.add_argument("-decay", "--lr_decay", help="Learning rate decay", action="store_true")
# parser.add_argument("-l", "--loss", help="Loss function.", default="CrossEntropy", choices=["MSE", "CrossEntropy"])
# parser.add_argument("-o", "--optimizer", default="Adam", choices=["Adam", "SGD"])
args = parser.parse_args()
setup = {"device": torch.device("cuda") if torch.cuda.is_available() else "cpu", "dtype": torch.float32}
print(f'Running on {setup["device"]}, PyTorch version {torch.__version__}')


def multi2binary(labels):
    return torch.tensor([0 if l < 2 or l > 7 else 1 for l in labels]).to(**setup)


def train():
    trainloader, testloader = dataloader(dataset=args.dataset, mode="train", index=-1, batchsize=args.batchsize,
                                         config=config)
    time = datetime.datetime.now()

    # Build up framework
    net = CNN6().to(**setup)
    loss_fn = nn.BCELoss()
    lr = args.learning_rate
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epoch//2.5, args.epoch//1.66],
                                                     gamma=0.1) if args.lr_decay else None
    writer = SummaryWriter(os.path.join(config['path_to_analysis'], 'train',
                                        f'{time:%d%b-%H%M}_{args.dataset}_{net.__name__()}_adam'))
    # make directory to save model
    if not os.path.isdir(config['path_to_model']):
        os.mkdir(config['path_to_model'])

    # Start training
    loss, ep = 0, 0
    try:
        for ep in range(args.epoch):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                labels = multi2binary(labels)
                inputs = inputs.to(**setup)
                pred = net(inputs)
                loss = loss_fn(input=pred.squeeze(), target=labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print(f'Epoch {ep+1:d}/{args.epoch}, Step {i+1:d}/{len(trainloader)}, Loss {loss.item():.3f}, Lr {lr}')
                    writer.add_scalar('training_loss', loss.item(), ep*len(trainloader)+i)
            if args.lr_decay:
                scheduler.step()
                lr = scheduler.get_last_lr()[0]

            torch.save({
                'epoch': ep+1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, os.path.join(config['path_to_model'], f'{net.__name__()}_adam_{args.dataset}_eps{ep+1}.pt'))

            # Evaluation
            # pr-curve
            classes = CLASSES[args.dataset.lower()]
            class_probs = []
            class_preds = []
            class_labels = []
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    labels = multi2binary(labels)
                    images = images.to(**setup)
                    output = net(images)
                    # class_probs_batch = [F.softmax(el, dim=0) for el in output]
                    # _, class_preds_batch = torch.max(output, 1)

                    class_probs_batch = output.reshape(labels.shape)
                    class_probs.append(class_probs_batch)
                    # class_preds.append(class_preds_batch)
                    class_labels.append(labels)

            # test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
            test_probs = torch.cat(class_probs)
            test_labels = torch.cat(class_labels)

            # for class_index in range(len(classes)):
            for class_index in range(2):
                # tensorboard_preds = test_preds == class_index
                tensorboard_labels = test_labels == class_index
                # tensorboard_prods = test_probs[:, class_index]
                tensorboard_prods = torch.abs(test_probs+class_index-1)

                # writer.add_pr_curve(classes[class_index], tensorboard_preds, tensorboard_prods,
                #                     global_step=ep*len(trainloader))
                writer.add_pr_curve(classes[class_index], tensorboard_labels, tensorboard_prods,
                                    global_step=ep*len(trainloader))
            # visualize parameters with histogram
            for m in net.state_dict():
                if 'weight' in m or 'bias' in m:
                    writer.add_histogram(m, net.state_dict()[m], global_step=ep*len(trainloader))

    except KeyboardInterrupt:
        torch.save({
                    'epoch': ep+1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
        }, os.path.join(config['path_to_model'], f'{net.__name__()}_adam_{args.dataset}_eps{ep+1}.pt'))

    finally:
        writer.close()


if __name__ == "__main__":
    train()
