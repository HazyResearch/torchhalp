import pytest
import numpy as np
import torch
from torch.autograd import Variable

from common import *

import sys
sys.path.append('..')
from optim import SVRG

from examples import regression

np.random.seed(0xdeadbeef)

#========================================================================================
# SVRG implementations
#========================================================================================

def baseline_svrg(x, y, w, lr, T=1, K=1, calc_gradient=None):
    for k in range(K):
        w_prev = w
        full_grad = calc_gradient(x, y, w, avg=True)
        print T
        for t in range(T):
            xi, yi = x[[t],:], y[t:t+1]
            print xi
            w_grad = calc_gradient(xi, yi, w)
            w_prev_grad = calc_gradient(xi, yi, w_prev)
            adjusted_grad = w_grad - w_prev_grad + full_grad
            w = w - (lr*adjusted_grad)
    return w

def pytorch_svrg(x, y, w, lr, T, K=1, n_features=None, n_classes=1):
    model = regression.utils.build_model(n_features, n_classes, initial_value=w)
    x = torch.from_numpy(x).float()
    # linear regression
    if n_classes == 1:
        y = torch.from_numpy(y).float().view(-1,1)
        loss = torch.nn.MSELoss()
    else: # multiclass logistic
        y = torch.from_numpy(y).long()
        loss = torch.nn.CrossEntropyLoss()

    synth_dataset = regression.utils.SynthDataset(x, y)
    train_loader = torch.utils.data.DataLoader(synth_dataset)

    svrg_opt = SVRG(model.parameters(), lr=lr, T=T, data_loader=train_loader)

    for k in range(K):
        for i, (data, target) in enumerate(train_loader):
            def closure(data=data, target=target):
                data = Variable(data, requires_grad=False)
                target = Variable(target, requires_grad=False)
                output = model(data)
                cost = loss(output, target)
                cost.backward()
                return cost
            svrg_opt.step(closure)
            w = np.asarray([p.data.numpy() for p in
                list(model.parameters())]).reshape(n_classes, n_features)
    return w


#========================================================================================
# Tests
#========================================================================================

@pytest.mark.parametrize("n_samples,n_features,lr,K",
[
    (1,   1,   1, 1),
    (1,   4, 0.1, 1),
    (1,   4, 0.1, 2),
    (10,  4, 0.1, 1),
    (10,  4, 0.1, 1),
    (10, 10, 0.1, 1),
    (10, 10, 0.5, 1),
    (10, 10, 0.5, 10)
])
def test_linear_regress(n_samples, n_features, lr, K):
    x = np.random.rand(n_samples, n_features)
    y = np.random.uniform(0,1, size=(n_samples,))
    w = np.random.uniform(0, 0.1, (1, n_features))

    np_value = baseline_svrg(x, y, w, lr, T=n_samples, K=K, calc_gradient=linear_grad)
    pytorch_value = pytorch_svrg(x, y, w, lr, T=n_samples, K=K, n_features=n_features)
    np.testing.assert_allclose(np_value, pytorch_value, rtol=1e-4)

@pytest.mark.parametrize("n_samples,n_features,n_classes,lr,K",
[
    (1, 1, 3,   1, 1),
    (1, 4, 3, 0.1, 1),
    (1, 4, 3, 0.1, 2),
    (2, 4, 3, 0.1, 1),
    (2, 4, 3, 0.5, 1),
    (2, 4, 3, 0.5, 2),
    (2, 4, 4, 0.5, 2),
])
def test_logistic_regress(n_samples, n_features, n_classes, lr, K):
    x = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, size=(n_samples,))
    w = np.random.uniform(0, 0.1, (n_classes, n_features))

    np_value = baseline_svrg(x, y, w, lr, T=n_samples, K=K, calc_gradient=logistic_grad)
    pytorch_value = pytorch_svrg(x, y, w, lr, T=n_samples, K=K, n_features=n_features,
                                 n_classes=n_classes)
    np.testing.assert_allclose(np_value, pytorch_value, rtol=1e-4)
