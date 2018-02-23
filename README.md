PyTorch SVRG
==============
This repo contains a PyTorch implementation of an SVRG optimizer. 

### Getting Started 

To run tests, run `pytest test_svrg.py -v`.

To add the SVRG optimizer to your existing PyTorch code:

- Import the optimizer 
`from svrg import SVRG`
- Change the optimizer to `optimizer = SVRG(model.parameters(), lr=args.lr, T=T, data_loader=train_loader)`
- Add a closure method which takes a datapoint and target, and recomputes the gradient. 

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
 
###  Notes

* The optimizer takes two additional arguments as compared to the SGD optimizer, `T` and `data_loader`. `T` indicates how often the full gradient over the entire dataset, a key step in the SVRG algorithm, is taken, where `T` is the number of weight updates in between updating the full gradient. The `data_loader` argument requires a PyTorch [DataLoader](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader), such that the gradient over the full dataset can be initiated internally in the optimizer.

* Currently, the optimizer doesn’t support multiple per-parameter options and parameter groups. 

* The following examples are provided in this repo using the SVRG optimizer: linear regression, logistic regression, and a small neural network on MNIST ([original PyTorch code with SGD](https://github.com/pytorch/examples/blob/master/mnist/main.py)).