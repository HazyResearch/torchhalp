'''Train CIFAR10 with PyTorch.'''

# modified from https://github.com/kuangliu/pytorch-cifar

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable

import csv

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
parser.add_argument('--T', type=int, help='T, only used for SVRG and HALP')
parser.add_argument('--mu', default=1, type=float, help='mu, only used for HALP')
parser.add_argument('--b', default=8, type=int, help='Number of bits to use, only used for HALP')
parser.add_argument('--num_epochs', default=1, type=int, help='Number of epochs')
parser.add_argument('--net', default='VGG', type=str, help='Network')
parser.add_argument('--opt', default='SGD', type=str, help='Optimizer for training')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--progress', action='store_true')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

if args.T is None and (args.opt == 'SVRG' or args.opt == 'HALP'):
    args.T = 2*len(trainloader) # Number of batches

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
else:
    print('==> Building model..')
    if args.net == 'VGG':
        net = VGG('VGG19')
    elif args.net == 'ResNet':
        net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
if args.opt == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.opt == 'SVRG':
    optimizer = SVRG(net.parameters(), lr=args.lr, data_loader=trainloader, T=args.T)
elif args.opt == 'HALP':
    optimizer = HALP(net.parameters(), lr=args.lr, data_loader=trainloader, T=args.T, mu=args.mu, bits=args.b)

# Training
def train(epoch):
    losses = []
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        def closure(data=data, target=target):
            data = Variable(data, requires_grad=False)
            target = Variable(target, requires_grad=False)

            # Need to pass an argument to use cuda or not
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = net(data)
            cost = criterion(output, target)
            cost.backward()
            return cost

        optimizer.zero_grad()
        loss = optimizer.step(closure)
        losses.append(loss.data[0])
        if args.progress:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f' % (loss.data[0]))
    return losses

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    accuracies = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        acc = 100.*correct/total
        if args.progress:
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), acc, correct, total))

    # Save checkpoint.
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.module if use_cuda else net,
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.{}'.format(args.opt))
    #     best_acc = acc

    accuracies.append(acc)
    return accuracies

# Create folders and files to save metrics
if not os.path.isdir('results'):
    os.mkdir('results')

if args.opt == 'SGD':
    if not os.path.isdir('results/sgd'):
        os.mkdir('results/sgd')
    tag = 'results/sgd/{}_lr_{}'.format(args.net, args.lr)
if args.opt == 'SVRG':
    if not os.path.isdir('results/svrg'):
        os.mkdir('results/svrg')
    tag = 'results/svrg/{}_lr_{}_T_{}'.format(args.net, args.lr, args.T)
if args.opt == 'HALP':
    if not os.path.isdir('results/halp'):
        os.mkdir('results/halp')
    tag = 'results/halp/{}_lr_{}_T{}_mu_{}_b_{}'.format(args.net, args.lr, args.T, args.mu, args.b)
training_file = '{}_train.csv'.format(tag)
test_file = '{}_test.csv'.format(tag)

# Remove file since we append to it over training
if start_epoch == 0:
    try:
        os.remove(training_file)
        os.remove(test_file)
    except OSError:
        pass

# Do training
for epoch in range(start_epoch, start_epoch+args.num_epochs):
    training_losses = train(epoch)
    test_accuracies = test(epoch)

    # Save metrics
    with open(training_file, 'a+') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in training_losses:
            csvwriter.writerow([row])
    with open(test_file, 'a+') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(test_accuracies)
