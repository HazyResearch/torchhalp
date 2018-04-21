# CIFAR-10 Example on ResNet-18

We can run SGD, SVRG, and HALP on a ResNet-18 with minimal modification to existing code (https://github.com/kuangliu/pytorch-cifar).

To run ResNet-18 on CIFAR10 using HALP:

`python main.py --num_epochs 150 --opt HALP --lr 0.1 --T 196 --mu 20 --progress`.

The learning rate is set to 0.1. The hyperparameter `T`, which determines how often we take the full gradient and do the bit centering is set to 196 batches (about half an epoch). `Mu`, a hyperparameter that affects the scaling of the low-precision model in HALP is set to 20. The default number of bits used for HALP is 8, but the `--b` flag can be set to use a different number of bits. Since this is simulation, any number of bits greater than 1 can be used.

As you decrease T, your code will likely take longer to run since you will be taking the full gradient more often. As you increase T, you may notice that your code runs faster but may converge slower. We explored sweeping different hyperparameters in our [results](results/ResNet18_results.ipynb) and found that with the settings above, we could get acceptable validation accuracy down to 4 bits.


