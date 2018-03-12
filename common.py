import numpy as np

def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x, axis=1).reshape((-1,1))
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1,1)

def logistic_grad(x, y, w, avg=False):
    """Compute the gradient for multi-class logistic regression"""
    xi_dot_w = np.dot(x, w.T)
    pred = stablesoftmax(xi_dot_w)
    for i in range(len(x)):
        pred[i][y[i]] = pred[i][y[i]] - 1
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

# https://github.com/pytorch/pytorch/blob/master/test/common.py
def iter_indices(tensor):
    if tensor.dim() == 0:
        return range(0)
    if tensor.dim() == 1:
        return range(tensor.size(0))
    return product(*(range(s) for s in tensor.size()))