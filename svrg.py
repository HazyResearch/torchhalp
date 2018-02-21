from torch.optim.optimizer import Optimizer, required
import torch
import numpy 
import copy 
from torch.autograd import Variable

#TODO(mleszczy): Be able to inherit from different optimizers 
class SVRG(torch.optim.SGD):
    r"""Implements stochastic variance reduction gradient descent (optionally with momentum).
    Args:
        params (iterable): iterable of parameters to optimize 
        lr (float): learning rate
        T (int): number of inner iterations, frequency of taking full gradient 
        data_loader (DataLoader): dataloader to use to load training data
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Example:
    .. note::
    """

    def __init__(self, params, lr=required, T=required, data_loader=required, weight_decay=0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SVRG, self).__init__(params, **defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SVRG doesn't support per-parameter options "
                             "(parameter groups)")

        # TODO(mleszczy): Add these to parameter group or state?
        self._params = self.param_groups[0]['params']
        self._params_prev = []
        self._full_grad = []

        for p in self._params:
            self._full_grad.append(torch.nn.Parameter(p.data, requires_grad=False))
            self._params_prev.append(torch.nn.Parameter(p.data.clone(), requires_grad=True))

        self.data_loader = data_loader
        self.state['t_iters'] = T
        self.T = T

    def __setstate__(self, state):
        super(SVRG, self).__setstate__(state)

    def step(self, closure=None, verbose=False):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None 

        # Calculate full gradient (step 1)
        if self.state['t_iters'] == self.T:
            # Reset gradients before accumulating them 
            for p in self._params:
                if p.grad is not None:
                    p.grad.detach()
                    p.grad.zero_()

            # Accumulate gradients
            for i, (data, target) in enumerate(self.data_loader):
                data, target = Variable(data), Variable(target)
                closure(self._params, data, target)
            
            # Adjust summed gradients by num_iterations accumulated over 
            for p in self._params: 
                p.grad.data /= len(self.data_loader)

            # Copy gradients over to full_grad
            for i, p in enumerate(self._params): 
                self._full_grad[i] = p.grad.data[:]
                # print self._full_grad[i]

            # Copy w to w_tilde
            for p, p0 in zip(self._params, self._params_prev):
                p0.data.copy_(p.data)

            # Reset t 
            self.state['t_iters'] = 0
            
            print self._full_grad[0].norm(2)

        # Calculate w_tilde gradient 
        for i, p in enumerate(self._params_prev):
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()
        closure(self._params_prev)

        # Calculate w gradient 
        for i, p in enumerate(self._params):
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()
        closure(self._params)

        # Update params
        for p, p0, fg in zip(self._params, self._params_prev, self._full_grad):
            d_p = p.grad.data
            d_p0 = p0.grad.data 

            # Adjust gradient in place (step 2)
            # print d_p.norm(2), d_p0.norm(2), fg.norm(2)
            d_p = d_p - d_p0 + fg 
            # Call optimizer update step (step 3)
            super(SVRG, self).step()
      
        self.state['t_iters'] += 1 

        # TODO(mleszczy): What do we want to return?
        return d_p.cpu().numpy()
