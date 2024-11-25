# ---------------------------------------------------------------------------- #
# Modified from ResNet, https://arxiv.org/pdf/1512.03385.pdf                   #
# See section 4.2 for the model architecture on CIFAR-10                       #
# ---------------------------------------------------------------------------- #
import os, sys
code_path = '../..'
data_path = '../../data/CIFAR10'
results_path = '.'

sys.path.append(code_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
import random
import numpy as np

import torchvision.transforms as transforms

import argparse
import csv

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--optimizer', type=str, default='alice', help='Specify optimizer: {sgdm, adam, adahessian, alice, powerlaw}')
parser.add_argument('--report_steps', type=int, default=100, help='Steps to take before evaluating test loss.')
parser.add_argument('--model', type=str, default='18', help='Use ResNet: {18, 34, 50, 101, 152}')
parser.add_argument('--d1', type=int, default=64, help='Channel dimensions in first block. Secondary blocks are multiplies of this.')
parser.add_argument('--d_max', type=int, default=512, help='Channel dimensions in first block. Secondary blocks are multiplies of this.')

# Optimization parameters
parser.add_argument('--learn_rate', type=float, default=5e-3, help='Learning rate and standard deviation of parameter perturbations.')
parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1 coefficient and momentum coefficient.')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2 coefficient.')
parser.add_argument('--eps', type=float, default=1E-8, help='Adam eps coefficient.')
parser.add_argument('--w1', type=float, default=0, help='Weight of L1 regularization.')
parser.add_argument('--w2', type=float, default=0, help='Weight of L2 regularization.')
parser.add_argument('--phi', type=float, default=None, help='Quasi-Newton step fraction.')
parser.add_argument('--omega', type=float, default=1., help='Look-ahead fraction of QN step.')
parser.add_argument('--limit_method', type=str, default='adam', help='Step limitation method.')
parser.add_argument('--lr_min', type=float, default=0., help='Exploration minimum learning rate.')
parser.add_argument('--lr_max', type=float, default=None, help='Maximum learning rate.')
parser.add_argument('--quick_steps', type=int, default=0, help='Quick steps taken between each full curvature update.')

parser.add_argument('--hess_comp', type=str, default='abs', help='Hessian computation type: zero, abs, rms.')
parser.add_argument('--grad_glass', action='store_true', help='Include gradient glass in optimization.')

parser.add_argument('--total_epochs', type=int, default=90, help='Number of epochs to train.')
parser.add_argument('--reduce_epochs', type=int, default=80, help='Number of epochs to reduce learning rate.')
parser.add_argument('--reduce_factor', type=float, default=0.1, help='Factor to reduce learning rate.')

parser.add_argument('--num_valid_data', type=int, default=0, help='Number of validation data')
parser.add_argument('--batch_size', type=int, default=100, help='batch_size of the train set')
parser.add_argument('--seed', type=int, default=0, help='random seed for the run.')
parser.add_argument('--log_prefix', type=str, default='R0T0S0', help='prefix for CSV files.')
args = parser.parse_args()

print('ResNet18 modified for curvature and method comparisons. Input Arguments: ', str(args))

powerlawcsv = f"{results_path}/{args.log_prefix}_pow.csv"
traincsv = f"{results_path}/{args.log_prefix}_trn.csv"
testcsv = f"{results_path}/{args.log_prefix}_tst.csv"

# Reproducibility
torch.set_float32_matmul_precision('high')
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# CIFAR-10 dataset
train_valid_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform, download=True)
train_dataset, valid_dataset = torch.utils.data.random_split(train_valid_dataset, [50000 - args.num_valid_data, args.num_valid_data])
test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
if args.num_valid_data > 0:
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True)
else:
    valid_loader = None
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

class Mask(torch.nn.Module):
    def __init__(self,
                 mode_in: int,
                 num_ch: int,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # Store logic format constants.
        self.mode_in = mode_in
        self.num_ch = num_ch

        # Create delta parameters, <1, num_ch, 1>.
        self.P = torch.nn.Parameter(torch.empty((1, self.num_ch, 1), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.constant_(self.P, 1.)

    def forward(self, X: Tensor) -> Tensor:
        # Save size of input X for reshaping output Y.
        X_size = torch.tensor(X.size(), requires_grad=False)
        if X_size[self.mode_in] != self.num_ch:
            raise ValueError('Input mode {} contains {} rather than {} elements.'.format(
                              self.mode_in, X_size[self.mode_in], self.num_ch))
        num_slow = torch.prod(X_size[:self.mode_in])
        num_fast = 1 if len(X_size)==self.mode_in+1 else torch.prod(X_size[self.mode_in+1:])
        Y = X.view((num_slow, self.num_ch, num_fast))*self.P
        return Y.view(list(X_size))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.mask1 = Mask(1, c_out) 
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.mask2 = Mask(1, c_out) 
        self.shortcut = nn.Sequential()
        self.mask3 = Mask(1, self.expansion*c_out) 
        if stride != 1 or c_in != self.expansion*c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, self.expansion*c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*c_out))

    def forward(self, x0):
        x1 = F.relu(self.mask1(self.bn1(self.conv1(x0))))
        x2 = self.bn2(F.relu(self.mask2(self.conv2(x1)) + self.shortcut(x0)))
        return self.mask3(x2)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, c_in, c_out, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv3 = nn.Conv2d(c_out, self.expansion*c_out, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*c_out)

        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != self.expansion*c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, self.expansion*c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*c_out))

    def forward(self, x0):
        x1 = F.relu(self.bn1(self.conv1(x0)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = self.bn3(F.relu(self.conv3(x2) + self.shortcut(x0)))
        return x3

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, d1=64, d_max=512):
        super(ResNet, self).__init__()
        self.c_in = d1

        self.conv1 = nn.Conv2d(3, d1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(d1)
        self.mask1 = Mask(1, d1)
        self.layer1 = self._make_layer(block, d1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*d1, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*d1, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*d1, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*d1*block.expansion, num_classes)

    def _make_layer(self, block, c_out, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.c_in, c_out, stride))
            self.c_in = c_out * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        y = F.relu(self.mask1(self.bn1(self.conv1(x))))
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = F.avg_pool2d(y, 4)
        y = y.view(y.size(0), -1)
        out = self.linear(y)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], d1=args.d1, d_max=args.d_max)

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], d1=args.d1, d_max=args.d_max)

def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3], d1=args.d1, d_max=args.d_max)

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3], d1=args.d1, d_max=args.d_max)

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3], d1=args.d1, d_max=args.d_max)

if args.model == '18':
    model = ResNet18()
elif args.model == '34':
    model = ResNet34()
elif args.model == '50':
    model = ResNet50()
elif args.model == '101':
    model = ResNet101()
elif args.model == '152':
    model = ResNet152()
model.to(device)

initial = set()
layer1 = set()
layer2 = set()
layer3 = set()
layer4 = set()
final = set()
for par_name, par in model.named_parameters():
    if 'linear' in par_name:
        final.add(par_name)
    elif 'layer1' in par_name:
        layer1.add(par_name)
    elif 'layer2' in par_name:
        layer2.add(par_name)
    elif 'layer3' in par_name:
        layer3.add(par_name)
    elif 'layer4' in par_name:
        layer4.add(par_name)
    else:
        initial.add(par_name)

param_dict = {name: par for name, par in model.named_parameters()}
optim_groups = [
        {'params': [param_dict[par_name] for par_name in sorted(list(initial))]},
        {'params': [param_dict[par_name] for par_name in sorted(list(layer1))]},
        {'params': [param_dict[par_name] for par_name in sorted(list(layer2))]},
        {'params': [param_dict[par_name] for par_name in sorted(list(layer3))]},
        {'params': [param_dict[par_name] for par_name in sorted(list(layer4))]},
        {'params': [param_dict[par_name] for par_name in sorted(list(final))]}]

if args.optimizer == 'alice':
    from alice import Alice
    optimizer = Alice(model.parameters(), lr=args.learn_rate, betas=(args.beta1, args.beta2), eps=args.eps,
                w1=args.w1, w2=args.w2, phi=args.phi, omega=args.omega, limit_method=args.limit_method, lr_min=args.lr_min, lr_max=args.lr_max,
                hess_comp=args.hess_comp, grad_glass=args.grad_glass, quick_steps=args.quick_steps)
    print(optimizer.state_str())
elif args.optimizer == 'powerlaw':
    from powerlaw import Powerlaw
    optimizer = Powerlaw(optim_groups, lr=args.learn_rate, betas=(args.beta1, args.beta2), eps=args.eps,
                w1=args.w1, w2=args.w2, phi=args.phi, omega=args.omega,
                hess_comp=args.hess_comp, grad_glass=args.grad_glass)
elif args.optimizer == 'adahessian':
    from pytorch_optimizer import AdaHessian
    optimizer = AdaHessian(model.parameters(), lr=args.learn_rate, betas=(args.beta1, args.beta2), eps=args.eps)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
elif args.optimizer == 'sgdm':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learn_rate, momentum=args.beta1)
else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}.")

# For updating learning rate
def update_lr(optimizer, lr):
    for grp in optimizer.param_groups:
        grp['lr'] = lr

# Loss function
criterion = nn.CrossEntropyLoss(reduction='mean')

# Test epoch
def test_epoch(loader, eval_mode=True):
    with torch.no_grad():
        test_loss = torch.tensor(0., device=device)
        test_acc = torch.tensor(0., device=device)
        test_count = torch.tensor(0, device=device)

        if eval_mode:
            model.eval()
        else:
            model.train()

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, max_pred = torch.max(outputs, 1)
            test_loss.add_(loss*labels.size(0))
            test_acc.add_((max_pred == labels).sum())
            test_count.add_(labels.size(0))
        
        test_loss.div_(test_count)
        test_acc.div_(test_count)
    return test_loss, test_acc

# Training epoch
def train_epoch(steps):
    train_loss = torch.tensor(0., device=device)
    train_acc = torch.tensor(0., device=device)
    train_count = torch.tensor(0, device=device)

    for i, (images, labels) in enumerate(train_loader):
        model.train()
        images = images.to(device)
        labels = labels.to(device)
        def model_func():
            return model(images)
        def loss_func(outputs):
            return criterion(outputs, labels)

        if args.optimizer == 'alice' or args.optimizer == "powerlaw":
            loss, outputs = optimizer.step((model_func, loss_func))
        elif args.optimizer == 'adahessian':
            outputs = model_func()
            loss = loss_func(outputs)
            optimizer.zero_grad()
            loss.backward(create_graph=True)
            optimizer.step()
        else:
            outputs = model_func()
            loss = loss_func(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        steps += 1

        _, max_pred = torch.max(outputs, 1)
        train_loss.add_(loss*labels.size(0))
        train_acc.add_((max_pred == labels).sum())
        train_count.add_(labels.size(0))

        if (i+1) % args.report_steps == 0:
            train_loss.div_(train_count)
            train_acc.div_(train_count)

            if not valid_loader == None:
                valid_loss, valid_acc = test_epoch(valid_loader)
            else:
                valid_loss = torch.tensor(0, device=device)
                valid_acc = torch.tensor(0, device=device)


            test_loss, test_acc = test_epoch(test_loader)

            train_row = [steps, train_loss.item(), train_acc.item()]
            test_row = [steps, test_loss.item(), test_acc.item()];
            with open(traincsv, "a") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(train_row)
            with open(testcsv, "a") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(test_row)

            if args.optimizer == 'powerlaw':
                rho, h1, vexp = optimizer.get_attributes()
                csvrow = [steps]
                for i, vexp_ in enumerate(vexp):
                    csvrow.append(vexp_.item())
                    print(f" vexp_{i}: {vexp_:.2f}", end="")
                with open(powerlawcsv, "a") as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(csvrow)
                    
            print(f"   step: {steps:05d}", end="")
            print(" (Trn,Vld,Tst)Loss: {:.4f}, {:.4f}, {:.4f}  ".format(train_loss, valid_loss, test_loss), end="")
            print(" (Trn,Vld,Tst)Acc: {:.4f}, {:.4f}, {:.4f}  ".format(train_acc, valid_acc, test_acc))

            # Reset training accumulators for next report.
            train_loss = torch.tensor(0., device=device)
            train_acc = torch.tensor(0., device=device)
            train_count = torch.tensor(0, device=device)
    return steps

train_row = ['steps', 'train_loss', 'train_acc']
test_row = ['steps', 'test_loss', 'test_acc'];
with open(traincsv, "a") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(train_row)
with open(testcsv, "a") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(test_row)

steps = 0
lr=args.learn_rate;
for epoch in range(args.total_epochs):
    if epoch == args.reduce_epochs:
        lr = lr*args.reduce_factor
        update_lr(optimizer, lr)
    print("Epoch {}/{}:".format(epoch, args.total_epochs))
    steps = train_epoch(steps)

