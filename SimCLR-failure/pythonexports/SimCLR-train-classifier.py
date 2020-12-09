#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from GetDataLoaders import get_dataloaders, get_short_dataloaders
from architectures.AlexNetFeatureModified import AlexNetFeature
from architectures.NonLinearClassifier import Classifier
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


use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
batch_size = 192
lr = 1e-3
LUT_lr = [(5, 0.1),(25, 0.02),(45, 0.0004),(65,0.00008)]
num_epochs = 200
momentum = 0.9
weight_decay = 1e-6
nesterov = True
num_classes = 200
loaders = get_dataloaders('imagenet', batch_size=batch_size, num_workers=2, unsupervised=False, simclr=False)


# In[4]:


feature_net = AlexNetFeature().to(device)
#load pretrained weights in feature_net
state_dict = torch.load("weights/AlexNet_Decoupling_Contrastive_SimCLR_Features.pth")
feature_net.load_state_dict(state_dict['featurenet'], strict=False)

feature_net.eval()
for param in feature_net.parameters():
    param.requires_grad = False


classifier_net = Classifier().to(device)
classifier_optimizer = optim.Adam(classifier_net.parameters(), lr=lr, weight_decay=weight_decay)
#feature_optimizer = optim.Adam(feature_net.parameters(), lr=lr, weight_decay=weight_decay)
Networks =   {'classifier':classifier_net, 'feature':feature_net}
Optimizers = {'classifier':classifier_optimizer} #'feature':feature_optimizer}

Criterions = {'CE': nn.CrossEntropyLoss()}


# In[5]:


classifier_net


# In[6]:


def train_validate(data_loader, epoch, train=True):
    
    mode = "Train" if train else "Valid"
    if train is True:
        #for key in Networks:
        Networks['classifier'].train()
    else:
        #for key in Networks:
        Networks['classifier'].eval()
    
    
    losses = []
    correct = 0
    
    overallloss = None
    
    
    start_time = time.time()
    tqdm_bar = tqdm(data_loader)
    batch_sizes = 0
    for batch_idx, batch in enumerate(tqdm_bar):
        data, targets = batch
        
        data, targets = data.to(device), targets.to(device)
        
        with torch.no_grad():
            features = Networks['feature'](data, ['conv5'])
        
        if train is False:
            with torch.no_grad():
                output =  Networks['classifier'](features)
        else:
            #features = Networks['feature'](data, ['conv5'])
            output = Networks['classifier'](features)
            
    
        loss_ce = Criterions['CE'](output, targets)
        

        if train is True:
            loss_ce.backward()
            Optimizers['classifier'].zero_grad()
            #Optimizers['feature'].zero_grad()
            Optimizers['classifier'].step()
            #Optimizers['feature'].step()
               
        losses.append(loss_ce.item())
        output = F.softmax(output, dim=1)
        pred = output.argmax(dim=1, keepdim=True).squeeze_(dim=1)
        #print(pred.size(), targets.size())
        correct_iter = pred.eq(targets.view_as(pred)).sum().item()
        correct += correct_iter
        batch_sizes += data.size(0)
        tqdm_bar.set_description('{} Epoch: [{}] Loss: CE {:.4f}, Correct: {}/{}'.format(mode, epoch, loss_ce.item(), correct, batch_sizes))
        
    
    end_time = time.time()
    print("Time for epoch pass {}".format(end_time-start_time))
    acc = float(correct/len(data_loader.dataset))
    averageloss = float(np.mean(losses))
    overallloss = {'ce':averageloss, 'acc':acc*100.0}
    print('{} set: Average loss: CE {:.4f}, Accuracy {}/{} {:.4f}%\n'.format(mode, overallloss['ce'], correct, len(data_loader.dataset), overallloss['acc']))
    return overallloss


# In[7]:


def run_main_loop(loaders, num_epochs):
    writer = SummaryWriter('./logs/AlexNet_SimCLR_NonLinearClassifier')
    save_path = "weights/AlexNet_Decoupling_Contrastive_SimCLR_NonLinearClassifier.pth"
    best_acc = 0
    for epoch in range(num_epochs):
        #print("Performing {}th epoch".format(epoch))
        train_loss = train_validate(loaders['train_loader'], epoch, train=True)
        val_loss = train_validate(loaders['valid_loader'], epoch, train=False)
        
        
        writer.add_scalar('CELoss/train', train_loss['ce'], epoch)
        writer.add_scalar('Accuracy/train', train_loss['acc'], epoch)
        writer.add_scalar('CELoss/Valid', val_loss['ce'], epoch)
        writer.add_scalar('Accuracy/Valid', val_loss['acc'], epoch)
        
        if val_loss['acc'] > best_acc :
            best_acc = val_loss['acc']
            #save model
            states = {
                'epoch': epoch + 1,
                'best_accuracy': best_acc
            }
            for key in Networks:
                states[key+"net"] = Networks[key].state_dict()
            for key in Optimizers:
                states[key+"optimizer"] = Optimizers[key].state_dict()
            torch.save(states, save_path)
            print('Model Saved')


# In[ ]:


run_main_loop(loaders, num_epochs)


# In[ ]:


test_loss = train_validate(loaders['test_loader'], 1, train=False)
print("Test Average Loss is {:.4f}, and Accuracy is {:.4f}".format(test_loss['ce'], test_loss['acc']*100.0))


# In[ ]:




