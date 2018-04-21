import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch import optim

import numpy as np
import argparse
from sklearn import linear_model, datasets

from utils import SynthDataset

from torchhalp.optim import SVRG, HALP

import matplotlib
matplotlib.use('pdf') # uncomment to run on raiders9
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Linear regression')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default= 0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--T', type=int, default=200, metavar='T',
                        help='how many iterations between taking full gradient')
    parser.add_argument('--mu', default=4, type=float,
                        help='mu, only used for HALP')
    parser.add_argument('--b', default=8, type=int,
                        help='Number of bits to use, only used for HALP')
    parser.add_argument('--n', type=int, default=100, metavar='NS',
                        help='number of samples')
    parser.add_argument('--num-features', type=int, default=5, metavar='F',
                        help='number of features')
    parser.add_argument('--sgd', action='store_true',
                        help='Runs stochastic gradient descent')
    parser.add_argument('--svrg', action='store_true',
                        help='Runs SVRG')
    parser.add_argument('--halp', action='store_true',
                        help='Runs HALP algorithm')
    parser.add_argument('--all', action='store_true',
                        help='Runs all optimizer algorithms')
    parser.add_argument('--save_graph', action='store_true',
                        help='Saves a graph of the results.')
    return parser.parse_args()

def add_plot(iters, dist, label, log_y=True, T=None):
    if log_y:
        plt.plot = plt.semilogy
    plt.figure(0)
    plt.plot(range(iters), dist, label=label)

# https://discuss.pytorch.org/t/adaptive-learning-rate/320/23
def step_decay_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

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
                                    n_features=n_features,
                                    random_state=0xc0ffee)

    # Solve for optimal solution
    w_opt, _, _, _= np.linalg.lstsq(X, Y, rcond=None)

    # Make dataloader
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float().view(-1,1)
    synth_dataset = SynthDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(synth_dataset, shuffle=True)

    loss = torch.nn.MSELoss()

    def build_model():
        # Create model
        model = torch.nn.Sequential()
        model.add_module("linear", torch.nn.Linear(n_features, 1, bias=False))
        model.linear.weight.data.fill_(0.0)
        if args.cuda:
            model.cuda()
        return model

    def train(optimizer, lr_decay=False):
        # Training
        dist_to_optimum = []
        iters = 0
        for e in range(num_epochs):

            for i, (data, target) in enumerate(train_loader):
                # We need to add this function to models when we want to use SVRG or HALP
                def closure(data=data, target=target):
                    data = Variable(data, requires_grad=False)
                    target = Variable(target, requires_grad=False)
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()
                    output = model(data)
                    cost = loss(output, target)
                    cost.backward()
                    return cost

                # We need to zero the optimizer for SGD
                # (already done internally for SVRG and HALP)
                optimizer.zero_grad()

                # This is the key line to perform the optimizer step
                # We don't need to call forward/backward explicitly (in addition to in the closure)
                # since the optimizer will call the closure
                optimizer.step(closure)

                # Performance metric: distance to optimum
                w = np.asarray([p.data.cpu().numpy() for p in list(model.parameters())])
                dist = np.linalg.norm(w-w_opt)
                dist_to_optimum.append(dist)
                if iters % n == 0:
                    print("Iteration = %d, Dist_to_opt = %s" % (iters , dist))
                iters += 1

            if lr_decay:
                step_decay_lr_scheduler(optimizer, e, lr_decay=0.1, lr_decay_epoch=200)

        return dist_to_optimum

    # Optimizer
    if args.sgd or args.all:
        model = build_model()
        opt = optim.SGD(model.parameters(), lr=args.lr)
        dist = train(opt, lr_decay=False)
        add_plot(num_epochs*len(train_loader), dist, label='SGD')

    if args.svrg or args.all:
        model = build_model()
        opt = SVRG(model.parameters(), T=T, data_loader=train_loader, lr=args.lr)
        dist = train(opt)
        add_plot(num_epochs*len(train_loader), dist, label='SVRG', T=T)

    if args.halp or args.all:
        model = build_model()
        opt = HALP(model.parameters(), T=T, data_loader=train_loader, lr=args.lr, mu=args.mu, bits=args.b)
        dist = train(opt)
        add_plot(num_epochs*len(train_loader), dist, label='HALP', T=T)

    if args.save_graph:
        plt.figure(0)
        plt.ylabel('Distance to Optimum')
        plt.xlabel('Iterations')
        plt.legend(loc='best')
        plt.savefig('results.svg')

if __name__ == "__main__":
    main()
