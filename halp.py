from torch.optim.optimizer import Optimizer, required
import torch
import copy, logging
from torch.autograd import Variable
import math

import quantize

# Monkey patch Torch.tensor to add quantization simulation
torch.Tensor.quantize_ = quantize.quantize_
torch.Tensor.saturate_ = quantize.saturate_
torch.cuda.FloatTensor.quantize_ = quantize.quantize_
torch.cuda.FloatTensor.saturate_ = quantize.saturate_

# NB: Note we choose the baseclass dynamically below.
class HALP(torch.optim.SGD):
    r"""Implements high-accuracy low-precision algorithm.
    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate
        T (int): number of iterations between the step to take the full grad/save w
        data_loader (DataLoader): dataloader to use to load training data
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    Example:
    .. note::
    """

    def __init__(self, params, lr=required, T=required, data_loader=required,
        weight_decay=0.0, opt=torch.optim.SGD, mu=1e-2, bits=8):

        defaults = dict(lr=lr, weight_decay=weight_decay)
        self.__class__ = type(self.__class__.__name__,
                              (opt,object),
                              dict(self.__class__.__dict__))
        # logging.info(f"Using base optimizer {opt} in SVRG")
        super(self.__class__, self).__init__(params, **defaults)

        if len(self.param_groups) != 1:
            raise ValueError("HALP doesn't support per-parameter options "
                             "(parameter groups)")

        params = self.param_groups[0]['params']
        self._params = params

        self._curr_w = [p.data for p in params]
        self._z = [p.data.clone() for p in params]
        self._prev_w = [p.data.clone() for p in params]

        # Gradients are lazily allocated and don't exist yet. However, gradients are
        # the same shape as the weights so we can still allocate buffers here
        self._curr_grad = [p.data.clone() for p in params]
        self._prev_grad = [p.data.clone() for p in params]
        self._full_grad = None

        self.data_loader = data_loader
        self.state['t_iters'] = T
        self.T = T # Needed to trigger full gradient
        # logging.info(f"Data Loader has {len(self.data_loader)} with batch {self.data_loader.batch_size}")

        self._scale_factors = [1 for p in params]
        self._bits = bits
        self._mu = mu

    def __setstate__(self, state):
        super(self.__class__, self).__setstate__(state)

    def _zero_grad(self):
        for p in self._params:
            if p.grad is not None:
                p.grad.detach()
                p.grad.zero_()

    def _set_weights_grad(self,ws,gs):
        for idx, p in enumerate(self._params):
            if ws is not None: p.data = ws[idx]
            if gs is not None and p.grad is not None:
                p.grad.data = gs[idx]
                assert (p.grad.data.data_ptr() == gs[idx].data_ptr())

    def _rescale(self):
        """Update scale for z"""
        div_factor = math.pow(2.0, self._bits-1) - 1
        for i, fg in enumerate(self._full_grad):
            self._scale_factors[i] = fg.norm() / (self._mu * div_factor)

    def _reset_z(self):
        """Set z to zero"""
        for p in self._z:
            p.fill_(0)

    def _recenter(self, ws):
        """ Add z to w """
        for w, z in zip(ws, self._z):
            w.add_(z)

    def _compute_full_grad(self, closure):
        # Setup the full grad
        # Reset gradients before accumulating them
        self._set_weights_grad(self._prev_w, self._full_grad)
        self._zero_grad()

        # Accumulate gradients
        for i, (data, target) in enumerate(self.data_loader):
            closure(data, target)

        # Adjust summed gradients by num_iterations accumulated over
        for p in self._params:
            if p.grad is not None:
                p.grad.data /= len(self.data_loader)

        # Since p.grad is dynamically allocated, the pointers to the gradients won't
        # be set before backward is called the first time
        if self._full_grad is None:
            self._full_grad = [p.grad.data.clone() for p in self._params]

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Calculate full gradient
        if self.state['t_iters'] == self.T:
            self._recenter(self._prev_w)
            self._compute_full_grad(closure)
            self._rescale()
            self._reset_z()
            # Reset t
            self.state['t_iters'] = 0

        # Setup the previous grad.
        self._set_weights_grad(self._prev_w, self._prev_grad)
        self._zero_grad()
        closure()

        # Calculate the current grad.
        self._set_weights_grad(self._curr_w, self._curr_grad)
        self._zero_grad()
        loss = closure()

        # Adjust the current gradient using the previous gradient and the full gradient.
        # We have normalized so that these are all comparable.
        for i, p in enumerate(self._params):
            # Adjust gradient in place
            if p.grad is not None:
                p.grad.data -= (self._prev_grad[i] - self._full_grad[i])
                p.grad.data.quantize_(self._scale_factors[i], self._bits)

        # Set weights to z for update
        self._set_weights_grad(self._z, None)
        # Call optimizer update step
        super(self.__class__, self).step()

        # Clamp weights to low precision representation for saturated add
        for p, sf in zip(self._z, self._scale_factors):
            p.saturate_(sf, self._bits)

        self.state['t_iters'] += 1

        # Set weights in params for user to access current w
        # Add z to prev_w to form curr_w
        for p, p0 in zip(self._curr_w, self._prev_w):
            p.copy_(p0)
        self._recenter(self._curr_w)
        self._set_weights_grad(self._curr_w, self._curr_grad)

        return loss