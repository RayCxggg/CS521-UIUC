#!/usr/bin/env python
# coding: utf-8

# # Set up for dataset and model
# 
# Package installation, loading, and dataloaders. There's also a resnet18 model defined.

# In[ ]:


# !pip install tensorboardX

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import datasets, transforms
# from tensorboardX import SummaryWriter

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

np.random.seed(42)
torch.manual_seed(42)


## Dataloaders
# train_dataset = datasets.CIFAR10('cifar10_data/', train=True, download=True, transform=transforms.Compose(
#     [transforms.ToTensor()]
# ))
test_dataset = datasets.CIFAR10('cifar10_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[ ]:


def tp_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return .5 * (x + delta) * (1 - ind1) * (1 - ind2) + x * ind2

def tp_smoothed_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return (x + delta) ** 2 / (4 * delta) * (1 - ind1) * (1 - ind2) + x * ind2

class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std

class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs
    
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1, activation='relu'):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.activation = activation
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def act_function(self, preact):
        if self.activation == 'relu':
            act = F.relu(preact)
        elif self.activation[:6] == '3prelu':
            act = tp_relu(preact, delta=float(self.activation.split('relu')[1]))
        elif self.activation[:8] == '3psmooth':
            act = tp_smoothed_relu(preact, delta=float(self.activation.split('smooth')[1]))
        else:
            assert self.activation[:8] == 'softplus'
            beta = int(self.activation.split('softplus')[1])
            act = F.softplus(preact, beta=beta)
        return act

    def forward(self, x):
        out = self.act_function(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x  # Important: using out instead of x
        out = self.conv1(out)
        out = self.conv2(self.act_function(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, cuda=True, half_prec=False,
        activation='relu', fts_before_bn=False, normal='none'):
        super(PreActResNet, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = 64
        self.avg_preact = None
        self.activation = activation
        self.fts_before_bn = fts_before_bn
        if normal == 'cifar10':
            self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1)
            self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)
            print('no input normalization')
        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()
        if half_prec:
            self.mu = self.mu.half()
            self.std = self.std.half()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, n_cls)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride, self.activation))
            # layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []

        out = self.normalize(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if return_features and self.fts_before_bn:
            return out.view(out.size(0), -1)
        out = F.relu(self.bn(out))
        if return_features:
            return out.view(out.size(0), -1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def PreActResNet18(n_cls, cuda=True, half_prec=False, activation='relu', fts_before_bn=False,
    normal='none'):
    #print('initializing PA RN-18 with act {}, normal {}'.format())
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, cuda=cuda, half_prec=half_prec,
        activation=activation, fts_before_bn=fts_before_bn, normal=normal)


# intialize the model
model = PreActResNet18(10, cuda=True, activation='softplus1').to(device)
model.eval()

# Hyperparameters
epochs = 50
learning_rate = 0.001

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training")):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# # Train the model
# best_acc = 0
# for epoch in range(epochs):
#     train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    
#     print(f"Epoch {epoch+1}/{epochs}")
#     print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
#     # Save the best model
#     if train_acc > best_acc:
#         best_acc = train_acc
#         torch.save(model.state_dict(), 'models/best_model.pth')
#         print(f"Model saved with accuracy: {best_acc:.2f}%")

# print(f"Training complete! Best accuracy: {best_acc:.2f}%")

# Load the best model for evaluation
model.load_state_dict(torch.load('models/best_model.pth'))



# # Implement the Attacks
# 
# Functions are given a simple useful signature that you can start with. Feel free to extend the signature as you see fit.
# 
# You may find it useful to create a 'batched' version of PGD that you can use to create the adversarial attack.

# In[7]:


def pgd_linf_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    adv_x.requires_grad_(True) 
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        # TODO: Calculate the loss
        loss = ce_loss(output, labels)
        loss.backward()
        # TODO: compute the adv_x
        # find delta, clamp with eps
        # Take a step in the direction of the sign of the gradient
        adv_x.data = adv_x.data + eps_step * adv_x.grad.data.sign()
        # Project back to the L_inf ball and clip to [0, 1]
        adv_x.data = torch.max(torch.min(adv_x.data, x + eps), x - eps)
        adv_x.data = torch.clamp(adv_x.data, 0, 1)
        adv_x.grad.zero_()
   
    return adv_x


# In[8]:


def pgd_l2_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    adv_x.requires_grad_(True) 
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        batch_size = x.size()[0]
        # TODO: Calculate the loss
        loss = ce_loss(output, labels)
        loss.backward()
        grad = adv_x.grad.data
        # TODO: compute the adv_x
        # find delta, clamp with eps, project delta to the l2 ball
        # HINT: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgdl2.py 
        # Normalize the gradient for L2 step
        grad_norm = torch.norm(grad.view(batch_size, -1), dim=1).view(batch_size, 1, 1, 1)
        grad_normed = grad / (grad_norm + 1e-8)
        adv_x.data = adv_x.data + eps_step * grad_normed

        # Project perturbation onto L2 ball of radius eps
        delta = adv_x.data - x
        delta_flat = delta.view(batch_size, -1)
        delta_norm = torch.norm(delta_flat, p=2, dim=1).view(batch_size, 1, 1, 1)
        factor = torch.clamp(eps / (delta_norm + 1e-8), max=1.0)
        delta = delta * factor
        adv_x.data = x + delta

        # Clamp to [0, 1]
        adv_x.data = torch.clamp(adv_x.data, 0, 1)
        adv_x.grad.zero_()
   
    return adv_x


# # Evaluate Single and Multi-Norm Robust Accuracy
# 
# In this section, we evaluate the model on the Linf and L2 attacks as well as union accuracy.

# In[9]:


def test_model_on_single_attack(model, attack='pgd_linf', eps=0.1):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        if attack == 'pgd_linf':
            # TODO: get x_adv untargeted pgd linf with eps, and eps_step=eps/4
            x_adv = pgd_linf_untargeted(model, x_batch, y_batch, k=10, eps=eps, eps_step=eps/4)
        elif attack == 'pgd_l2':
            # TODO: get x_adv untargeted pgd l2 with eps, and eps_step=eps/4
            x_adv = pgd_l2_untargeted(model, x_batch, y_batch, k=10, eps=eps, eps_step=eps/4)
        else:
            pass
        
        # get the testing accuracy and update tot_test and tot_acc
        # tot_acc += TODO
        # tot_test += TODO

        with torch.no_grad():
            outputs = model(x_adv)
            preds = torch.argmax(outputs, dim=1)
            tot_acc += (preds == y_batch).sum().item()
            tot_test += y_batch.size(0)
            
    print('Robust accuracy %.5lf' % (tot_acc/tot_test), f'on {attack} attack with eps = {eps}')

# ## Multi-Norm Robust Accuracy

# In[14]:


def test_model_on_multi_attacks(model, eps_linf=8./255., eps_l2=0.75):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        # TODO: get x_adv_linf and x_adv_l2 untargeted pgd linf and l2 with eps, and eps_step=eps/4
        x_adv_linf = pgd_linf_untargeted(model, x_batch, y_batch, k=10, eps=eps_linf, eps_step=eps_linf/4)
        x_adv_l2 = pgd_l2_untargeted(model, x_batch, y_batch, k=10, eps=eps_l2, eps_step=eps_l2/4)
        
        ## calculate union accuracy: correct only if both attacks are correct
        
        out = model(x_adv_linf)
        pred_linf = torch.max(out, dim=1)[1]
        out = model(x_adv_l2)
        pred_l2 = torch.max(out, dim=1)[1]
        
        # TODO: get the testing accuracy with multi-norm robustness and update tot_test and tot_acc
        correct = ((pred_linf == y_batch) & (pred_l2 == y_batch)).sum().item()
        tot_acc += correct
        tot_test += y_batch.size(0)
            
    print('Robust accuracy %.5lf' % (tot_acc/tot_test), f'on multi attacks')

# Test the best model on single norm attacks (Linf and L2)
print("\nTesting best model on single norm attacks:")
test_model_on_single_attack(model, attack='pgd_linf', eps=8./255.)
test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)

# Test the best model on multi-norm attacks
print("\nTesting best model on multi-norm attacks:")
test_model_on_multi_attacks(model, eps_linf=8./255., eps_l2=0.75)

# Compare with pre-trained models
print("\nComparing with pre-trained models:")

# Model 1 (Linf-trained)
print("\nTesting Linf-trained model:")
model.load_state_dict(torch.load('models/pretr_Linf.pth'))
test_model_on_single_attack(model, attack='pgd_linf', eps=8./255.)
test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)
test_model_on_multi_attacks(model, eps_linf=8./255., eps_l2=0.75)

# Model 2 (L2-trained)
print("\nTesting L2-trained model:")
model.load_state_dict(torch.load('models/pretr_L2.pth'))
test_model_on_single_attack(model, attack='pgd_linf', eps=8./255.)
test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)
test_model_on_multi_attacks(model, eps_linf=8./255., eps_l2=0.75)

# Model 3 (RAMP-trained for multi-norm)
print("\nTesting RAMP-trained model:")
model.load_state_dict(torch.load('models/pretr_RAMP.pth'))
test_model_on_single_attack(model, attack='pgd_linf', eps=8./255.)
test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)
test_model_on_multi_attacks(model, eps_linf=8./255., eps_l2=0.75)