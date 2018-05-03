High-Accuracy Low-Precision Training
====================================
This repo contains a PyTorch implementation of the HALP optimizer from the paper [High-Accuracy Low-Precision Training](https://arxiv.org/abs/1803.03383) as well as a full-precision SVRG optimizer. It is designed for explanatory purposes rather than high-performance.

## Getting Started

```
git clone git@github.com:HazyResearch/torchhalp.git && cd torchhalp
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python setup.py install
pytest test/ -v
```

This only supports PyTorch version 0.3.1 or lower.

## Use in Other PyTorch Code
To add the optimizers to your existing PyTorch code:

1. Import the optimizer
`from torchhalp.optim import HALP`
2. Change the optimizer to `optimizer = HALP(model.parameters(), lr=args.lr, T=T, data_loader=train_loader)`
3. Add a closure method which takes a datapoint and target, and recomputes the gradient.
```
def closure(data=data, target=target):
	data = Variable(data, requires_grad=False)
	target = Variable(target, requires_grad=False)
    if cuda:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    return loss
 ```
4. Pass the closure method to the step function when you call `optimizer.step(closure)`.

## Examples

We include examples for [linear regression](examples/regression) and [ResNet-18 on CIFAR-10](examples/cifar10).

##  Notes

* This is meant to be a simulation to evaluate the effect of HALP on accuracy, but as a simulation, this implementation adds overhead with quantization.

* The SVRG and HALP optimizers take two additional arguments as compared to the SGD optimizer, `T` and `data_loader`. `T` indicates how often the full gradient over the entire dataset, a key step in the SVRG algorithm, is taken, where `T` is the number of batches in between updating the full gradient. The `data_loader` argument requires a PyTorch [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), such that the gradient over the full dataset can be initiated internally in the optimizer. The HALP optimizer has the additional arguments of `mu`, `bits`, and `unbiased` which affect the quantization, where `mu` contributes to the dynamic rescaling, `bits` is the number of bits used for the quantized numbers, and `unbiased` indicates whether stochastic rounding is used.

* Currently, the SVRG and HALP optimizers don’t support multiple per-parameter options and parameter groups.

* Stateful LSTMs are not supported due to the optimizer's self-contained nature. However, we can still use learned hidden layers or stateless LSTMs.
