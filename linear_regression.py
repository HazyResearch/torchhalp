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
    n = 100
    n_features = 4
    num_epochs = 100
    T = n 

    torch.manual_seed(6)
    np.random.seed(6)

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

    loss = torch.nn.MSELoss(size_average=True)
    svrg = SVRG(model.parameters(), T=T, data_loader=train_loader, lr=1e-4)
    batch_size = 1

    dist_to_optimum = []
    iters = 0 
    for e in range(num_epochs):
        for i, (data, target) in enumerate(train_loader):  
            data = Variable(data, requires_grad=False)
            target = Variable(target, requires_grad=False)
            def closure(parameters=None, data=data, target=target): 
                param_num= 0 
                for child in model.children():
                    for name, p in child.named_parameters():
                        child.register_parameter(name, parameters[param_num])
                        param_num+=1
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