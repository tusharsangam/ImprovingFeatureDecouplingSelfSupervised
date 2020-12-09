#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from GetDataLoaders import get_dataloaders, get_short_dataloaders
from architectures.Resnets import resnet50_cifar as resnet50
from architectures.ContrastiveLoss import ContrastiveLoss
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import time
from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR


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
batch_size = 32
lr = 1e-3

num_epochs = 200
'''
momentum = 0.9
nesterov = True
Lambdas = {'CE':1.0, 'MSE':1.0, 'NCE':1.0}
LUT_lr = [(90,0.01), (130,0.001), (190,0.0001), (210,0.00001), (230,0.0001), (245,0.00001)]
'''
weight_decay = 1e-6
loaders = get_dataloaders('imagenet', batch_size=batch_size, num_workers=2, unsupervised=True, simclr=True)
tau = 0.1
gamma = 2
decay_lr = 1e-6
accumulation_steps = 4 


# In[4]:


'''
import matplotlib.pyplot as plt
import numpy as np
import torchvision

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(loaders['train_loader'])
imagesi, imagesj = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(imagesi))
imshow(torchvision.utils.make_grid(imagesj))
'''


# In[5]:


feature_net = resnet50(128).to(device)
optimizer = optim.Adam(feature_net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = ExponentialLR(optimizer, gamma=decay_lr)


Networks = {'feature':feature_net}
Optimizers = {'feature':optimizer}

ContrastiveCriterion = ContrastiveLoss(tau=tau, normalize=True)
Criterions = {'CE': nn.CrossEntropyLoss(reduction='none'), 'MSE':nn.MSELoss() }


# In[6]:


feature_net


# In[7]:


def train_validate(data_loader, epoch, train=True):
    
    mode = "Train" if train else "Valid"
    if train is True:
        for key in Networks:
            Networks[key].train()
            Networks[key].zero_grad()
    else:
        for key in Networks:
            Networks[key].eval()
    
    
    losses = {'mse':[], 'nce':[]}
    
    
    overallloss = None
    
    
    start_time = time.time()
    tqdm_bar = tqdm(data_loader)
    
    for batch_idx, batch in enumerate(tqdm_bar):
        datai, dataj = batch
        
        datai, dataj = datai.to(device), dataj.to(device)
        if train is False:
            with torch.no_grad():
                _, featuresi = Networks['feature'](datai)
                _, featuresj = Networks['feature'](dataj)
                loss_nce = ContrastiveCriterion(featuresi, featuresj)
        else:
            _, featuresi = Networks['feature'](datai)
            _, featuresj = Networks['feature'](dataj)
            loss_nce = ContrastiveCriterion(featuresi, featuresj)
    
        loss_nce = loss_nce / accumulation_steps

        if train is True:
            loss_nce.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                for key in Optimizers:
                    Optimizers[key].step()
                    Networks[key].zero_grad()

        #calculate rotation invariance by MSE
        with torch.no_grad():
            features_mean = featuresi + featuresj 
            features_mean = torch.mul(features_mean, 0.5)
            loss_mse = Criterions['MSE'](featuresi, features_mean)
            loss_mse += Criterions['MSE'](featuresj, features_mean)   
            
        losses['mse'].append(loss_mse.item())
        losses['nce'].append(loss_nce.item())
        tqdm_bar.set_description('{} Epoch: [{}] Loss: NCE {:.4f}, MSE {:.4f}'.format(mode, epoch, loss_nce.item(), loss_mse.item()))
    
    
    end_time = time.time()
    print("Time for epoch pass {}".format(end_time-start_time))
    overallloss = {'mse': float(np.mean(losses['mse'])), 'nce':float(np.mean(losses['nce']))}
    print('{} set: Average loss: MSE {:.4f}, NT-Xent {:.4f}\n'.format(mode, overallloss['mse'], overallloss['nce']))
    return overallloss



def run_main_loop(loaders, num_epochs, starting_epoch=1):
    writer = SummaryWriter('./logs/AlexNet_SimCLR')
    save_path = "weights/AlexNet_Decoupling_Contrastive_SimCLR.pth"
    best_loss = np.Inf
    for epoch in range(starting_epoch, starting_epoch+num_epochs):
        
        
        train_loss = train_validate(loaders['train_loader'], epoch, train=True)
        val_loss = train_validate(loaders['valid_loader'], epoch, train=False)
        scheduler.step()
        
        writer.add_scalar('MSELoss/train', train_loss['mse'], epoch)
        writer.add_scalar('NT-XENTLoss/train', train_loss['nce'], epoch)
        writer.add_scalar('MSELoss/Valid', val_loss['mse'], epoch)
        writer.add_scalar('NT-XENTLoss/Valid', val_loss['nce'], epoch)
        writer.add_scalar('LR', Optimizers['feature'].param_groups[0]['lr'], epoch+1)
        
        
        
        if (epoch)%10 == 0 :
            best_loss = val_loss['nce']
            #save model
            states = {
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'scheduler': scheduler.state_dict()
            }
            for key in Networks:
                states[key+"net"] = Networks[key].state_dict()
            for key in Optimizers:
                states[key+"optimizer"] = Optimizers[key].state_dict()
            torch.save(states, save_path)
            print('Model Saved')


# In[8]:


run_main_loop(loaders, num_epochs)


# In[ ]:


save_path = "weights/AlexNet_Decoupling_Contrastive_SimCLR_Features.pth"
states = {
                'epoch':200,
                'scheduler': scheduler.state_dict()
            }
for key in Networks:
    states[key+"net"] = Networks[key].state_dict()
for key in Optimizers:
    states[key+"optimizer"] = Optimizers[key].state_dict()
torch.save(states, save_path)


# In[ ]:




