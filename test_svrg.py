from svrg import SVRG
from logistic_regression import build_model, SynthDataset

import numpy as np 
import torch
from torch.autograd import Variable

np.random.seed(0xdeadbeef)

#========================================================================================
# Helpers
#========================================================================================

def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=1).reshape((-1,1))
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1,1)

def logistic_grad(x, y, w, avg=False):
    """Compute the gradient for multi-class logistic regression"""
    xi_dot_w = np.dot(x, w.T)
    pred = stablesoftmax(xi_dot_w)
    pred[0][y] = pred[0][y] - 1
    grad = np.dot(pred.T, x)
    if avg: 
        grad = grad / float(len(x))
    return grad

def linear_grad(x, y, w, avg=False):
    """Compute the gradient for linear regression"""
    xi_dot_w = np.dot(x, w.T)
    grad = 2*np.dot(xi_dot_w.T - y.T, x)
    if avg: 
        grad = grad / float(len(x))
    return grad


#========================================================================================
# SVRG implementations 
#========================================================================================

def baseline_svrg(x, y, w, lr, T=1, K=1, calc_gradient=None):
    for k in range(K):
        w_prev = w
        full_grad = calc_gradient(x, y, w, avg=True)
        for t in range(T):
            xi, yi = x[[t],:], y[t:t+1]
            w_grad = calc_gradient(xi, yi, w)
            w_prev_grad = calc_gradient(xi, yi, w_prev)
            adjusted_grad = w_grad - w_prev_grad + full_grad
            w = w - (lr*adjusted_grad) 
            print w
    return w 

def pytorch_svrg(x, y, w, lr, T, K=1, n_features=None, n_classes=1):
    model = build_model(n_features, n_classes, initial_value=w)
    x = torch.from_numpy(x).float()
    # linear regression 
    if n_classes == 1:
        y = torch.from_numpy(y).float().view(-1,1)
        loss = torch.nn.MSELoss()
    else: # multiclass logistic
        y = torch.from_numpy(y).long()
        loss = torch.nn.CrossEntropyLoss()

    synth_dataset = SynthDataset(x, y)
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
            print w
    return w


#========================================================================================
# Tests 
#========================================================================================

# TODO(mleszczy): add sweeps 

def test_linear_regress(n_samples=4, n_features=5, lr=0.1, K=2):
    x = np.random.rand(n_samples, n_features)
    y = np.random.uniform(0,1, size=(n_samples,))
    w = np.random.uniform(0, 0.1, (1, n_features))

    np_value = baseline_svrg(x, y, w, lr, T=n_samples, K=K, calc_gradient=linear_grad)
    pytorch_value = pytorch_svrg(x, y, w, lr, T=n_samples, K=K, n_features=n_features)
    np.testing.assert_allclose(np_value, pytorch_value, rtol=1e-4)

def test_logistic_regress(n_samples=4, n_features=5, n_classes=3, lr=0.1, K=2):
    x = np.random.rand(1, n_features)
    y = np.random.randint(0, n_classes, size=(1,))
    w = np.random.uniform(0, 0.1, (n_classes, n_features))

    np_value = baseline_svrg(x, y, w, lr, K=K, calc_gradient=logistic_grad)
    pytorch_value = pytorch_svrg(x, y, w, lr, T=n_samples, K=K, n_features=n_features, 
                                 n_classes=n_classes)
    np.testing.assert_allclose(np_value, pytorch_value, rtol=1e-4)
