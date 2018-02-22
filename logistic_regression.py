import torch
from torch.autograd import Variable
import torch.utils.data as data
from torch import optim
from svrg import SVRG
import numpy as np
import random
# import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import copy 

import torch
from torch.autograd import Variable
from torch.nn import Parameter
from torch import optim
torch.set_printoptions(precision=10)

cuda = False

def build_model(input_dim, output_dim=1, initial_value=None):
    model = torch.nn.Sequential()
    module = torch.nn.Linear(input_dim, output_dim, bias=False)
    if initial_value is not None: 
        module.weight.data = torch.from_numpy(initial_value).type(torch.FloatTensor)
        model.add_module("linear", module)
    else:
        model.add_module("linear", torch.nn.Linear(input_dim, output_dim, bias=False))
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
    n_classes = 3
    iters = n*10
    K = iters / (n)
    T = n
    torch.manual_seed(6)
    np.random.seed(6)
    X,Y = datasets.make_classification(n_samples=n, 
                            n_features=n_features, 
                            n_informative=2, 
                            n_redundant=0,
                            n_repeated=0, 
                            n_classes=n_classes, 
                            n_clusters_per_class=1)

    clf = linear_model.LogisticRegression(fit_intercept=False, 
        multi_class='multinomial', 
        solver='sag', 
        C=1e100, 
        penalty='l2', 
        max_iter=10000, tol=1e-8)
    clf.fit(X, Y)
    w_opt = clf.coef_
    loss = torch.nn.CrossEntropyLoss(size_average=True)

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()
    synth_dataset = SynthDataset(X, Y)

    # Can also change batch_size of dataloader here 
    train_loader = torch.utils.data.DataLoader(synth_dataset)

    w = np.random.uniform(0, 0.1, (n_classes, n_features))

    model = build_model(n_features, n_classes, initial_value=w)

    # TODO(mleszczy): Add full cuda support 
    if cuda: 
        model.cuda() 
        model_prev.cuda() 
        loss = loss.cuda()

    svrg = SVRG(model.parameters(), lr=0.1, T=T, data_loader=train_loader)
    dist_to_optimum = [] 

    num_epochs = 100
    iters = 0
    for e in range(num_epochs):
        for i, (data, target) in enumerate(train_loader): 
            data = Variable(data, requires_grad=False)
            target = Variable(target, requires_grad=False)

            # This closure method would have to be copied into any program that wants to 
            # use SVRG :/ 
            def closure(data=data, target=target):
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
    # plt.plot(range(iters), dist_to_optimum, label="SVRG")
    # plt.ylabel('Distance to Optimum')
    # plt.xlabel('Iterations')
    # plt.legend(loc='best')
    # plt.show()

if __name__ == "__main__":
    main()