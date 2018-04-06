import numpy as np
from sklearn import linear_model, datasets
import argparse

import torch.utils.data as data
import torch
from torch.autograd import Variable
from torch import optim

from utils import SynthDataset

import sys
sys.path.append("../..")
from optim import SVRG, HALP

def parse_args():
    parser = argparse.ArgumentParser(description='Linear regression')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default= 0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--T', type=int, default=200, metavar='T',
                        help='how many iterations between taking full gradient')
    parser.add_argument('--n', type=int, default=100, metavar='NS',
                        help='number of samples')
    parser.add_argument('--num-features', type=int, default=4, metavar='F',
                        help='number of features')
    parser.add_argument('--opt', default='SGD', type=str, help='Optimizer for training')
    return parser.parse_args()

def main():
    args = parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    n = args.n
    n_features = args.num_features
    num_epochs = args.epochs
    T = args.T

    # Make synthetic dataset with sklearn
    X, Y = datasets.make_regression(n_samples=n,
                                    n_features=n_features)

    # Solve for optimal solution
    w_opt, _, _, _= np.linalg.lstsq(X, Y, rcond=None)

    # Make dataloader
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float().view(-1,1)
    synth_dataset = SynthDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(synth_dataset)

    # Create model
    model = torch.nn.Sequential()
    model.add_module("linear", torch.nn.Linear(n_features, 1, bias=False))
    model.linear.weight.data.fill_(0.0)
    if args.cuda:
        model.cuda()

    loss = torch.nn.MSELoss(size_average=True)

    # Optimizer
    if args.opt == 'SGD':
        opt = optim.SGD(model.parameters(), lr=args.lr)

    elif args.opt == 'SVRG':
        opt = SVRG(model.parameters(), T=T, data_loader=train_loader, lr=args.lr)

    elif args.opt == 'HALP':
        opt = HALP(model.parameters(), T=T, data_loader=train_loader, lr=args.lr)

    # Training
    dist_to_optimum = []
    iters = 0
    for e in range(num_epochs):
        for i, (data, target) in enumerate(train_loader):

            # We need to add this function to models when we want to use SVRG
            def closure(data=data, target=target):
                data = Variable(data, requires_grad=False)
                target = Variable(target, requires_grad=False)
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                cost = loss(output, target)
                cost.backward()
                return cost

            # This is the key line to perform the optimizer step
            # We don't need to call forward/backward explicitly (in addition to in the closure)
            # since the optimizer will call the closure
            opt.step(closure)

            # Performance metric: distance to optimum
            w = np.asarray([p.data.cpu().numpy() for p in list(model.parameters())])
            dist = np.linalg.norm(w-w_opt)
            dist_to_optimum.append(dist)
            if iters % T == 0:
                print("Iteration = %d, Dist_to_opt = %s" % (iters , dist))
            iters += 1

if __name__ == "__main__":
    main()