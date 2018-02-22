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
        params = self.param_groups[0]['params']
        
        self._params = params

        self._curr_w = [p.data for p in params]
        self._prev_w = [p.data.clone() for p in params] 

        # Gradients are lazily allocated and don't exist yet. However, gradients are 
        # the same shape as the weights so we can still allocate buffers here 
        self._prev_grad = [p.data.clone() for p in params]
        self._full_grad = [p.data.clone() for p in params] 

        self.data_loader = data_loader
        self.state['t_iters'] = T
        self.T = T

    def __setstate__(self, state):
        super(SVRG, self).__setstate__(state)

    # This is only changing the pointer to data and not copying data 
    def _switch_weights_to_copy(self, copy_w):
        for (w_new, p) in zip(copy_w, self._params):
            p.data = w_new

    # This is actually copying data (setting pointers to grad.data didn't work)
    def _copy_grads_from_params(self, grad_buffer):
        for (grad_data, p) in zip(grad_buffer, self._params):
            grad_data.copy_(p.grad.data)

    def _zero_grad(self):
        for p in self._params:
                if p.grad is not None:
                    p.grad.detach()
                    p.grad.zero_()

    def step(self, closure=None, verbose=False):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # Calculate full gradient 
        if self.state['t_iters'] == self.T:
            # Reset gradients before accumulating them 
            self._zero_grad()

            # Accumulate gradients
            for i, (data, target) in enumerate(self.data_loader):
                data, target = Variable(data), Variable(target)
                closure(data, target)
            
            # Adjust summed gradients by num_iterations accumulated over 
            for p in self._params: 
                p.grad.data /= len(self.data_loader)

            self._copy_grads_from_params(self._full_grad)
                
            # Copy w to prev_w
            for p, p0 in zip(self._curr_w, self._prev_w):
                p0.copy_(p)

            # Reset t 
            self.state['t_iters'] = 0
            
        # Copy prev_w over to parameters
        self._switch_weights_to_copy(self._prev_w)
        self._zero_grad()
        # Calculate prev_w gradient 
        closure()
        self._copy_grads_from_params(self._prev_grad)

         # Copy w over to parameters
        self._switch_weights_to_copy(self._curr_w)
        self._zero_grad()
        # Calculate w gradient 
        closure()
        # We don't need to copy out these gradients

        for p, d_p0, fg in zip(self._params, self._prev_grad, self._full_grad):
            d_p = p.grad.data

            # Adjust gradient in place 
            d_p -= (d_p0 - fg) 

            # Call optimizer update step 
            super(SVRG, self).step()
      
        self.state['t_iters'] += 1 

        # Return w
        # TODO(mleszczy): Add support for multiple layers 
        return p.data.cpu().numpy()
