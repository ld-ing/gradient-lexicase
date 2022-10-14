# Modified from: https://github.com/kuangliu/pytorch-cifar

'''Training image classification on CIFAR-10/100 and SVHN
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

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='Training baseline image classification')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--arch', default=0, type=int, help='arch index')
parser.add_argument('--dataset', default='C10', type=str, help='use C10, C100 or SVHN')
parser.add_argument('--seed', default=666, type=int)
parser.add_argument('--save', action='store_true', help='save checkpoint')
args = parser.parse_args()

torch.random.manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

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
        trainset, batch_size=128, shuffle=True, num_workers=2)

    selectset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_test)
    selectloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

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


n_epoch = 200

# Model
print('==> Building model..')
net = [VGG, ResNet18, ResNet50, DenseNet121, MobileNetV2, SENet18, EfficientNetB0][args.arch]
net = net(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
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
            
    return 100.*correct/total


# training
for epoch in range(start_epoch, start_epoch+n_epoch):
    train(epoch)
    acc = test(epoch)
    scheduler.step()

print(args.arch, acc)

# Save checkpoint.
if args.save:
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    save_dir = 'ckpt_base_{}_{}_{}_{}'.format(args.dataset, args.arch, args.seed, acc)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, save_dir+'/ckpt.pth')
    print('Checkpoint saved to:', save_dir)
