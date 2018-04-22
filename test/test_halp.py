import pytest
import numpy as np
import torch
from torch.autograd import Variable

from utils import *

from torchhalp.optim import HALP
from examples import regression

np.random.seed(0xdeadbeef)

#========================================================================================
# Helpers
#========================================================================================

def quantize(vect, b, scale_factor, biased=False):
    if not biased:
        random_vect = np.random.uniform(0, 1, size=vect.shape)
        vect = np.floor((vect/float(scale_factor)) + random_vect)
    else:
        vect = np.floor(vect/float(scale_factor) + 0.5)
    min_value = -1 * (2**(b-1))
    max_value = 2**(b-1) - 1
    vect = np.clip(vect, min_value, max_value)
    return vect

def dequantize(vect, scale_factor):
    return vect*scale_factor

#========================================================================================
# HALP implementations
#========================================================================================

def baseline_halp(x, y, w, lr, b, mu, n, T=1, K=1, calc_gradient=None):
    s_k = 1.0 # s_k needs an initial value to complete w addition
    z = np.zeros(w.shape)
    iters = 0
    for k in range(K):
        for idx in range(n):
            if iters % T == 0:
                # Recenter
                w = w + z # w is full precision
                g_k = calc_gradient(x, y, w, avg=True)
                # Rescale
                s_k = float(np.linalg.norm(g_k)) / (mu * (2**(b-1) - 1))
                z = np.zeros(w.shape)
            xi, yi = x[[idx],:], y[idx:idx+1]
            z = z - (lr*(calc_gradient(xi, yi, w + z) - calc_gradient(xi, yi, w) + g_k))
            z = quantize(z, b, s_k, biased=True)
            z = dequantize(z, s_k)
            iters += 1
    return w + z


def pytorch_halp(x, y, w, lr, b, mu, T=1, K=1, n_features=None, n_classes=1):
    model = regression.utils.build_model(n_features, n_classes, initial_value=w)
    x = torch.from_numpy(x).float()
    # Linear regression
    if n_classes == 1:
        y = torch.from_numpy(y).float().view(-1,1)
        loss = torch.nn.MSELoss()
    else: # Multiclass logistic
        y = torch.from_numpy(y).long()
        loss = torch.nn.CrossEntropyLoss()

    synth_dataset = regression.utils.SynthDataset(x, y)
    train_loader = torch.utils.data.DataLoader(synth_dataset)
    halp_opt = HALP(model.parameters(), lr=lr, T=T, data_loader=train_loader, bits=b, mu=mu, biased=True)

    for k in range(K):
        for i, (data, target) in enumerate(train_loader):
            def closure(data=data, target=target):
                data = Variable(data, requires_grad=False)
                target = Variable(target, requires_grad=False)
                output = model(data)
                cost = loss(output, target)
                cost.backward()
                return cost
            halp_opt.step(closure)
            w = np.asarray([p.data.numpy() for p in
                list(model.parameters())]).reshape(n_classes, n_features)
    return w


#========================================================================================
# Tests
#========================================================================================

@pytest.mark.parametrize("n_samples,n_features,lr,K,b,mu,T",
[
    (1,   1,   1,  1,  8,   1, 1),
    (1,   4, 0.1,  1,  8,   1, 1),
    (1,   4, 0.1,  4,  8,   1, 2),
    (10,  4, 0.1,  1,  8,   1, 10),
    (10,  4, 0.1,  1,  8,   1, 10),
    (10, 10, 0.1,  1,  8,   1, 10),
    (10, 10, 0.5,  1,  8,   1, 10),
    (10, 10, 0.5, 10,  8,   1, 10),
    (10, 10, 0.5, 10,  8, 0.1, 10),
    (10, 10, 0.5, 10, 16, 0.1, 10),
    (5,  10, 0.5, 10, 16,   1, 5),
    (10,  4, 0.1,  1,  8,   1, 20),
    (10,  4, 0.1,  1,  8,   1, 20),
    (10, 10, 0.1,  1,  8,   1, 20),
    (10, 10, 0.5,  1,  8,   1, 20),
    (10, 10, 0.5, 10,  8,   1, 20),
    (10, 10, 0.5, 10,  8, 0.1, 20),
    (10, 10, 0.5, 10, 16, 0.1, 20)
])
def test_linear_regress(n_samples, n_features, lr, K, b, mu, T):
    x = np.random.rand(n_samples, n_features)
    y = np.random.uniform(0,1, size=(n_samples,))
    w = np.random.uniform(0,0.1, (1, n_features))
    np_value = baseline_halp(x, y, w, lr, b, mu, n=n_samples, T=T, K=K, calc_gradient=linear_grad)
    pytorch_value = pytorch_halp(x, y, w, lr, b, mu, T=T, K=K, n_features=n_features)
    np.testing.assert_allclose(np_value, pytorch_value, rtol=1e-4)

@pytest.mark.parametrize("n_samples,n_features,n_classes,lr,K,b,mu",
[
    (1, 1, 3,   1, 1, 8,   1),
    (1, 4, 3, 0.1, 1, 8,   1),
    (1, 4, 3, 0.1, 2, 8,   1),
    (2, 4, 3, 0.1, 1, 8,   1),
    (2, 4, 3, 0.5, 1, 8,   1),
    (2, 4, 3, 0.5, 2, 8,   1),
    (2, 4, 4, 0.5, 2, 8,   1),
    (2, 4, 4, 0.5, 2, 8, 0.1),
    (2, 4, 4, 0.5, 2, 16,0.1),
    (2, 4, 4, 0.5, 2, 16,  1)

])
def test_logistic_regress(n_samples, n_features, n_classes, lr, K, b, mu):
    x = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, size=(n_samples,))
    w = np.random.uniform(0, 0.1, (n_classes, n_features))
    np_value = baseline_halp(x, y, w, lr, b, mu, n=n_samples, T=n_samples, K=K, calc_gradient=logistic_grad)
    pytorch_value = pytorch_halp(x, y, w, lr, b, mu, T=n_samples, K=K, n_features=n_features,
                                 n_classes=n_classes)
    np.testing.assert_allclose(np_value, pytorch_value, rtol=1e-4)
