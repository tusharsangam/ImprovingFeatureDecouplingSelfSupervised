#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from GetDataLoaders import get_dataloaders, get_short_dataloaders
from architectures.AlexNetFeatureModified import AlexNetFeature
from architectures.TransferLearningNet import Flatten
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import time
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm


# In[2]:


'''
# we skip the probs for now
gama = 2.0
with open(os.path.join("./PUprobs", 'prob.dat'), 'r') as file_input:
    train_prob_str = file_input.readlines()
    train_prob = [float(i_prob_str.rstrip('\n')) for i_prob_str in train_prob_str]
    print(len(train_prob)/4.0)
    train_weight = [1.0 if 0==i%4 else 1-train_prob[i]**gama for i in range(len(train_prob))]
'''


# In[3]:


class Classifier(nn.Module):
    def __init__(self, nChannels=256, num_classes=200, pool_size=6, pool_type='max'):
        super(Classifier, self).__init__()
        nChannelsAll = nChannels * pool_size * pool_size

        layers = []
        if pool_type == 'max':
            layers.append(nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            #layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        elif pool_type == 'avg':
            layer.append(nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        layers.append(nn.BatchNorm2d(nChannels, affine=False))
        layers.append(Flatten())
        layers.append(nn.Linear(nChannelsAll, num_classes))
        self.classifier = nn.Sequential(*layers)
        self.initilize()
    
    def forward(self, feat):
        return self.classifier(feat)
    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fin = m.in_features
                fout = m.out_features
                std_val = np.sqrt(2.0/fout)
                m.weight.data.normal_(0.0, std_val)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


# In[4]:


use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
batch_size = 192
lr = 0.1
LUT_lr = [(5, 0.01),(25, 0.002),(45, 0.0004),(65,0.00008)]
num_epochs = 65
momentum = 0.9
weight_decay = 5e-4
nesterov = True
num_classes = 200
loaders = get_dataloaders('imagenet', batch_size=batch_size, num_workers=2, unsupervised=False, simclr=False)


# In[5]:


feature_net = AlexNetFeature().to(device)
#load pretrained weights in feature_net
state_dict = torch.load("weights/AlexNet_Decoupling_Contrastive_SimCLR_Features.pth")
feature_net.load_state_dict(state_dict['featurenet'])

feature_net.eval()
for param in feature_net.parameters():
    param.requires_grad = False

classifier_net = Classifier().to(device)
classifier_optimizer = optim.SGD(classifier_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
Networks =   {'classifier':classifier_net}
Optimizers = {'classifier':classifier_optimizer}

Criterions = {'CE': nn.CrossEntropyLoss()}


# In[6]:


def adjust_learning_rates(epoch):
    # filter out the networks that are not trainable and that do
    # not have a learning rate Look Up Table (LUT_lr) in their optim_params
    lr = next((lr for (max_epoch, lr) in LUT_lr if max_epoch>epoch), LUT_lr[-1][1])
    for key in Optimizers:
        for g in Optimizers[key].param_groups:
            prev_lr = g['lr']
            if prev_lr != lr:
                print("Learning Rate Updated from {} to {}".format(prev_lr, lr))
                g['lr'] = lr

def train_step(batch, train=True):
    data, targets = batch
    
    if train is True:
        for key in Optimizers:
            Optimizers[key].zero_grad()
    
    #to cuda
    
    data, targets = data.to(device), targets.to(device)
   
    
    #collect features
    with torch.no_grad():
        features = feature_net(data, ['conv5'])
    
    if train is False:
        with torch.no_grad():
            pred = Networks['classifier'](features)
            #calculate loss
            loss_cls =  Criterions['CE'](pred, targets)
            
    else:
        pred = Networks['classifier'](features)
        #calculate loss
        loss_cls =  Criterions['CE'](pred, targets)
    
    if train is True:
        loss_cls.backward()
        for key in Optimizers:
            Optimizers[key].step()
    
    #calculate classification accuracy
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=1, keepdim=True)
    correct = pred.eq(targets.view_as(pred)).sum().item()
   
    
    return loss_cls.item(), correct
    


# In[7]:


def train_validate(data_loader, epoch, train=True):
    mode = "Train" if train else "Valid"
    if train is True:
        for key in Networks:
            Networks[key].train()
    else:
        for key in Networks:
            Networks[key].eval()
    
    losses = []
    correct = 0
    
    #if train:
        #adjust_learning_rates(epoch)
    
    start_time = time.time()
    
    tqdm_bar = tqdm(data_loader)
    total_number = 0
    for batch_idx, sample in enumerate(tqdm_bar):
        
        loss, correct_step = train_step(sample, train=train)
        losses.append(loss)
        correct += correct_step
        total_number += sample[0].size(0)
        tqdm_bar.set_description('{} Epoch: {} Loss: {:.6f}, Accuracy: {}/{} [{:.4f}%]'.format(mode, epoch, loss, correct, total_number, 100.0*(correct/total_number)))
    
    end_time = time.time()
    print("Time for epoch pass {}".format(end_time-start_time))
    
    loss = float(np.mean(losses))
    acc =  float(correct / len(data_loader.dataset))
    print('{} set: Average loss: {:.4f}, Accuracy:{}/{} ({:.4f}%)\n'.format(mode, loss, correct, len(data_loader.dataset), 100.0*acc))
    return loss, acc


def run_main_loop(loaders, num_epochs):
    writer = SummaryWriter('logs/Imagenet-Classification-Without-Finetuning-SimCLR')
    save_path = "weights/AlexNet_Decoupling_Contrastive_SimCLR_Classifier.pth"
    best_acc = 0
    for epoch in range(num_epochs):
        #print("Performing {}th epoch".format(epoch))
        
        train_loss, train_acc = train_validate(loaders['train_loader'], epoch, train=True)
        val_loss, val_acc = train_validate(loaders['valid_loader'], epoch, train=False)
    
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/Valid', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/Valid', val_acc, epoch)
    
        
        if val_acc > best_acc  :
            best_acc = val_acc
            #save model
            states = {
                'epoch': epoch + 1,
                'best_accuracy': best_acc
            }
            for key in Networks:
                states[key+"model"] = Networks[key].state_dict()
            for key in Optimizers:
                states[key+"optimizer"] = Optimizers[key].state_dict()
            torch.save(states, save_path)
            print('Model Saved')


# In[ ]:


run_main_loop(loaders, num_epochs)


# In[ ]:


loss, acc = train_validate(loaders['test_loader'], 1, train=False)


# In[ ]:




