import torch
from torch.autograd import Variable
from torch import optim
from svrg import SVRG
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import torch.utils.data as data

import torch
from torch.autograd import Variable
from torch import optim
import argparse 

def build_model(n_features):
    model = torch.nn.Sequential()
    model.add_module("linear", torch.nn.Linear(n_features, 1, bias=False))
    model.linear.weight.data.fill_(0.0)
    return model

class SynthDataset(data.Dataset):
    def __init__(self, data, labels):
        self.data = data 
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main():
    parser = argparse.ArgumentParser(description='Logistic regression')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    n = 100
    n_features = 4
    num_epochs = 1000
    T = n 

    np.random.seed(args.seed)
    X, Y = datasets.make_regression(n_samples=n,
                                    n_features=n_features,
                                    noise=10,
                                    n_informative=1)
    w_opt, _, _, _= np.linalg.lstsq(X, Y)

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float().view(-1,1) 
    synth_dataset = SynthDataset(X, Y)
    train_loader = torch.utils.data.DataLoader(synth_dataset)

    model = build_model(n_features)
    if args.cuda:
        model.cuda()

    loss = torch.nn.MSELoss(size_average=True)
    svrg = SVRG(model.parameters(), T=T, data_loader=train_loader, lr=args.lr)
    batch_size = 1

    dist_to_optimum = []
    iters = 0 
    for e in range(args.epochs):
        for i, (data, target) in enumerate(train_loader):  
            # Need to add this function 
            def closure(data=data, target=target): 
                data = Variable(data, requires_grad=False)
                target = Variable(target, requires_grad=False)
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                cost = loss(output, target)
                cost.backward()
                return cost

            w = svrg.step(closure)
            dist = np.linalg.norm(w-w_opt)
            dist_to_optimum.append(dist)
            if iters % T == 0:
                print("Iteration = %d, Dist_to_opt = %s" % (iters , dist))
            iters += 1 

    print dist_to_optimum[-1]

if __name__ == "__main__":
    main()