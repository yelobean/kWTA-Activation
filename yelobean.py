
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import json
import time
import sys
import copy
import argparse
import os
import datetime

from kWTA import models
from kWTA import activation
from kWTA import attack
from kWTA import training
from kWTA import utilities
from kWTA import densenet
from kWTA import resnet
from kWTA import wideresnet

parser = argparse.ArgumentParser(description='PyTorch Training')

# parser.add_argument('--attack', '-a', default='', help='which attack? option:(blank, fgsm, bim, cw, pgd)')
parser.add_argument('--gpu', '-g', default='9', help='which gpu to use')
parser.add_argument('--which_AT', '-at', default='nat', help='which AT (nat, at, trades)')
parser.add_argument('--is_kWTA', '-wta', type=bool, default='', help='option WTA on/off')
parser.add_argument('--is_Wide', '-wide', type=bool, default='', help='option WIDE on/off')
parser.add_argument('--is_test', '-test', type=bool, default='T', help='only test')
parser.add_argument('--iters', '-i', default=10, type=int, help='how many attack iters')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

norm_mean = 0
norm_var = 1
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
])
cifar_train = datasets.CIFAR10("/home/yelobean/dataset", train=True, download=True, transform=transform_train)
cifar_test = datasets.CIFAR10("/home/yelobean/dataset", train=False, download=True, transform=transform_test)
train_loader = DataLoader(cifar_train, batch_size = 256, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size = 100, shuffle=True)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = torch.device('cuda:0')
name = ''
if args.is_Wide:
    name = name + '_wide'
if args.is_kWTA:
    name = name + '_kWTA'
name = name + '_' + args.which_AT
name = name + '_iter' + str(args.iters)

if args.is_Wide:
    if args.is_kWTA:
        model = wideresnet.SparseWideResNet(depth=34, num_classes=10, widen_factor=10, sp=0.1, sp_func='vol').to(device)
    else:
        model = wideresnet.WideResNet(depth=34, num_classes=10, widen_factor=10).to(device)
else:
    if args.is_kWTA:
        model = resnet.SparseResNet18(sparsities=[0.1, 0.1, 0.1, 0.1], sparse_func='vol').to(device)
    else:
        model = resnet.ResNet18().to(device)

if len(args.gpu) > 2:
    import torch.backends.cudnn as cudnn
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
eps = 8/255
for ep in range(200):
    print('gpu:', args.gpu)
    print('iter:', args.iters)
    print('is_kWTA:', args.is_kWTA)
    print('which_AT:', args.which_AT)
    print('is_Wide:', args.is_Wide)

    if ep == 100:
        for param_group in opt.param_groups:
                param_group['lr'] = 0.01
    if ep == 150:
        for param_group in opt.param_groups:
                param_group['lr'] = 0.001

    start_time = time.time()

    if not args.is_test:
        if args.which_AT == 'at':
            train_err, train_loss = training.epoch_adversarial(train_loader, model, opt=opt,
                                                               attack=attack.pgd_linf_untargeted, device=device, num_iter=args.iters,
                                                               epsilon=eps, randomize=True, alpha=2/255)
        elif args.which_AT == 'trades':
            train_err, train_loss = training.epoch_trade(train_loader, model, opt=opt, device=device,
                                                               num_iter=args.iters, epsilon=eps, alpha=2/255, beta=6.0)
        elif args.which_AT == 'nat':
            train_err, train_loss = training.epoch(train_loader, model, opt, device=device)
        else:
            raise print('AT name ERROR!')
    else:
        model.load_state_dict(torch.load('models/resnet18_cifar' + name + '.pth', map_location=device))
        train_err, train_loss = training.epoch(train_loader, model, device=device)


    times = str(datetime.timedelta(seconds=time.time() - start_time))
    print('Tr time:', times)

    start_time = time.time()
    test_err, test_loss = training.epoch(test_loader, model, device=device)
    times = str(datetime.timedelta(seconds=time.time() - start_time))
    print('Te time:', times)

    start_time = time.time()
    adv_err, adv_loss = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20,
        epsilon=eps, randomize=True, alpha=2/255)
    times = str(datetime.timedelta(seconds=time.time() - start_time))
    print('Adv Te time:', times)
    print('epoch', ep, 'train err', train_err, 'test err', test_err, 'adv_err', adv_err)

    if args.is_test:
        break
    torch.save(model.state_dict(), 'models/resnet18_cifar' + name + '_epoch200.pth')