# Modified from: https://github.com/kuangliu/pytorch-cifar

'''Training image classification with gradient lexicase selection on CIFAR-10/100 and SVHN
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from copy import deepcopy

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='Training gradient lexicase selection')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--arch', default=0, type=int, help='arch index')
parser.add_argument('--dataset', default='C10', type=str, help='use C10, C100 or SVHN')
parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--pop', default='4', type=int, help='population size')
parser.add_argument('--save', action='store_true', help='save checkpoint')
args = parser.parse_args()

torch.random.manual_seed(args.seed)
np.random.seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0
pop_size = args.pop

# Data
print('==> Preparing data..')
if args.dataset == "C10":
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

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2)

    selectset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test)
    selectloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)

    num_classes = 10

elif args.dataset == "C100":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2)

    selectset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_test)
    selectloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)

    num_classes = 100

elif args.dataset == "SVHN":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.SVHN(
        root='./data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2)

    selectset = torchvision.datasets.SVHN(
        root='./data', split='train', download=True, transform=transform_test)
    selectloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)

    testset = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)

    num_classes = 10
    

# Model
print('==> Building model..')

criterion = nn.CrossEntropyLoss()
n_epoch = 200

# get the Cosine Annealing learning rate
def CosineAnnealingLR(epoch, lr = args.lr, T_max=n_epoch*(pop_size+1), eta_min=0):  # assuming 0-indexed epoch
    return eta_min + 0.5*(lr - eta_min)*(1+np.cos((epoch)/T_max*np.pi))


# Training
def train(epoch, children):
    print('\nEpoch: %d' % epoch)
    optimizers = []
    for net in children:
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=CosineAnnealingLR(epoch),
                      momentum=0.9, weight_decay=1e-4)
        optimizers.append(optimizer)

    train_loss = [0. for _ in range(pop_size)]
    correct = [0. for _ in range(pop_size)]
    total = [0. for _ in range(pop_size)]
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        train_idx = batch_idx % pop_size
        optimizers[train_idx].zero_grad()
        outputs = children[train_idx](inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizers[train_idx].step()

        train_loss[train_idx] += loss.item()*targets.size(0)
        _, predicted = outputs.max(1)
        total[train_idx] += targets.size(0)
        correct[train_idx] += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss[train_idx]/total[train_idx], 100.*correct[train_idx]/total[train_idx], correct[train_idx], total[train_idx]))
        
    return children


def select(children):
    for net in children:
        net.eval()
    selection = list(range(pop_size))
    selected = np.random.randint(0, pop_size)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(selectloader):
            inputs, targets = inputs.to(device), targets.to(device)
            for child_idx in selection:
                outputs = children[child_idx](inputs)
                _, predicted = outputs.max(1)
                correct = predicted.eq(targets).sum().item()
                if correct / targets.size(0) < 1.:
                    selection.remove(child_idx)
            
            if len(selection) == 1:
                selected = selection[0]

            if len(selection) <= 1:
                print('selection process over after', batch_idx, "batches.")
                break
    
    # apply parent weights to all children
    parent_weights = deepcopy(children[selected].state_dict())
    for net in children:
            net.load_state_dict(parent_weights)

    return children


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total

    return acc

arch = [VGG, ResNet18, ResNet50, DenseNet121, MobileNetV2, SENet18, EfficientNetB0][args.arch]

# initialize children
children = [arch(num_classes=num_classes).to(device) for _ in range(pop_size)]

# main training loop
for epoch in range(start_epoch, start_epoch+n_epoch*(pop_size+1)):
    children = train(epoch, children)
    children = select(children)

    if (epoch+1)%50 == 0:
        acc = test(epoch, children[0])

        # Save checkpoint.
        if args.save:
            print('Saving..')
            state = {
                'net': children[0].state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            save_dir = 'ckpt_lexi_{}_{}_{}_{}'.format(args.dataset, args.arch, args.seed, acc)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            torch.save(state, save_dir+'/ckpt.pth')
            print('Checkpoint saved to:', save_dir)

print(args.arch, acc)
