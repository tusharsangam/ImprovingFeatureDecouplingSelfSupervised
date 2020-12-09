#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from GetDataLoaders import get_dataloaders, get_short_dataloaders
from architectures.AlexNetFeature import AlexNetFeature
from architectures.AlexNetClassifierModified import AlexNetClassifier
from architectures.LinearTransformationNormModified import LinearTransformationNorm, Normalize
from architectures.ContrastiveLoss import ContrastiveLoss
#from architectures.NCEAverage import NCEAverage
#from architectures.NCELoss import NCELoss
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import time
from torch import optim


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
lr = 0.1
LUT_lr = [(90,0.01), (130,0.001), (190,0.0001), (210,0.00001), (230,0.0001), (245,0.00001)]
num_epochs = 245
momentum = 0.9
weight_decay = 5e-4
nesterov = True
Lambdas = {'CE':1.0, 'MSE':1.0, 'NCE':1.0}

loaders = get_dataloaders('imagenet', batch_size=batch_size, num_workers=2)
ndata_train = len(loaders['train_loader'].dataset)
ndata_valid = len(loaders['valid_loader'].dataset)
t = 0.07
m = 4096
gamma = 2


# In[4]:


ndata_train, ndata_valid


# In[5]:


#from torch.optim.lr_scheduler import ExponentialLR

feature_net = AlexNetFeature().to(device)
classifier_net = AlexNetClassifier().to(device)
transformation_net = LinearTransformationNorm().to(device)

feature_optimizer = optim.SGD(feature_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
classifier_optimizer = optim.SGD(classifier_net.parameters() ,lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
transformation_optimizer = optim.SGD(transformation_net.parameters() ,lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

#Schedulers = {'feature':ExponentialLR(feature_optimizer, gamma=1e-6), 'classifier':ExponentialLR(classifier_optimizer, gamma=1e-6), 'transformation':ExponentialLR(transformation_optimizer, gamma=1e-6)}

Networks = {'feature':feature_net, 'classifier':classifier_net, 'transformation':transformation_net}
Optimizers = {'feature':feature_optimizer, 'classifier':classifier_optimizer, 'transformation':transformation_optimizer}

ContrastiveCriterion = ContrastiveLoss(tau=0.1)
#NCE = {'NCEAverage_train': NCEAverage(outputSize=ndata_train, K=m, T=t).cuda(), 'NCEAverage_valid': NCEAverage(outputSize=ndata_valid, K=m, T=t).cuda() , 'NCECriterion_train': NCELoss(ndata_train), 'NCECriterion_valid': NCELoss(ndata_valid)}
Criterions = {'CE': nn.CrossEntropyLoss(reduction='none'), 'MSE':nn.MSELoss() }


# In[6]:


#for sample in loaders['train_loader']:
    #print(sample[0].size(), sample[1].size(), sample[2].size())


# In[7]:


def adjust_learning_rates(epoch):
    # filter out the networks that are not trainable and that do
    # not have a learning rate Look Up Table (LUT_lr) in their optim_params
    lr = next((lr for (max_epoch, lr) in LUT_lr if max_epoch>epoch), LUT_lr[-1][1])
    for key in Optimizers:
        for g in Optimizers[key].param_groups:
            g['lr'] = lr

def AlexNetDecoupling(batch_idx, batch, PUWeights=None, train=True, accumulation_steps=4):
    data, targets, indices = batch
    
    if train is True:
        Optimizers['feature'].zero_grad()
        Optimizers['classifier'].zero_grad()
        Optimizers['transformation'].zero_grad()
    
    #to cuda
    
    data = data.to(device)
    targets = targets.to(device)
    indices = indices.to(device)

    
    #perform rotations & adjust data shape
    data_90 = torch.flip(torch.transpose(data,2,3),[2])
    data_180 = torch.flip(torch.flip(data,[2]),[3])
    data_270 = torch.transpose(torch.flip(data,[2]),2,3)
    data = torch.stack([data, data_90, data_180, data_270], dim=1)
    batch_size, rotations, channels, height, width = data.size()
    data = data.view([batch_size*rotations, channels, height, width])
    
    #debug for backward
    data.requires_grad = False 
    targets.requires_grad = False
    indices.requires_grad = False
    
    #collect features
    features = Networks['feature'](data)
    features_rot, features_invariance = torch.split(features, 2048, dim=1)
    
    #collect rotation prediction
    pred = Networks['classifier'](features_rot)
    
    
    #average features across 4 rotations
    features_invariance_instance = features_invariance[0::4,:] + features_invariance[1::4,:] + features_invariance[2::4,:] + features_invariance[3::4,:]
    features_invariance_instance = torch.mul(features_invariance_instance, 0.25) #fbar 192x2048
    
    #downsample to 128 & perform normalization of vector
    
    features_128_norm = Networks['transformation'](features_invariance_instance)#192x128
    features_128_norm_0 = Networks['transformation'](features_invariance[0::4,:])
    features_128_norm_90 = Networks['transformation'](features_invariance[1::4,:])
    features_128_norm_180 = Networks['transformation'](features_invariance[2::4,:])
    features_128_norm_270 = Networks['transformation'](features_invariance[3::4,:])
    
    features_128_list = [features_128_norm_0, features_128_norm_90, features_128_norm_180, features_128_norm_270]
    
    with torch.no_grad():
        #stack 192x2048 4 times to be  = 192x4x2048 = 768x2048
        features_invariance_instance_mean = torch.unsqueeze(features_invariance_instance,1).expand(-1,4,-1).clone()
        features_invariance_instance_mean = features_invariance_instance_mean.view(4*len(features_invariance_instance), 2048) #2048
    
    #calculate rotation loss ignore PU for now
    loss_cls_each = Criterions['CE'](pred, targets)
    loss_cls = torch.sum(loss_cls_each)/loss_cls_each.shape[0]
    
    #calculate rotation invariance by MSE
    with torch.no_grad():
        loss_mse = Criterions['MSE'](features_invariance, features_invariance_instance_mean)
    
    #calculate instance loss using NT-xent
    loss_nce = 0.0
    loss_nce = ContrastiveCriterion(features_128_norm_0, features_128_norm) + ContrastiveCriterion(features_128_norm_90, features_128_norm) + ContrastiveCriterion(features_128_norm_180, features_128_norm) + ContrastiveCriterion(features_128_norm_270, features_128_norm)
    
    loss_total = Lambdas['CE']*loss_cls + Lambdas['NCE']*loss_nce
    
    if train is True:
        loss_total.backward()
        Optimizers['feature'].step()
        Optimizers['classifier'].step()
        Optimizers['transformation'].step()
            
    
    #calculate classification accuracy
    pred = pred.argmax(dim=1, keepdim=True)
    correct = pred.eq(targets.view_as(pred)).sum().item()
    
    losses = {'ce':loss_cls.item(), 'mse':loss_mse.item(), 'nce':loss_nce.item(), 'correct':correct}
    
    return losses
    


# In[8]:


def train(data_loader, epoch, log_interval=50):
    
    Networks['feature'].train()
    Networks['classifier'].train()
    Networks['transformation'].train()
    
    losses = {'ce':[], 'mse':[], 'nce':[]}
    correct = 0
    train_loss = np.Inf
    train_acc = 0.0
    
    start_time = time.time()
    for batch_idx, sample in enumerate(data_loader):
        
        lossesdict = AlexNetDecoupling(batch_idx, sample, train=True)
        
        losses['ce'].append(lossesdict['ce'])
        losses['mse'].append(lossesdict['mse'])
        losses['nce'].append(lossesdict['nce'])
        correct += lossesdict['correct']
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: CE {:.6f}, MSE {:.6f}, NT-Xent {:.6f}'.format(epoch, batch_idx*len(sample[0]), len(data_loader.dataset), 100. * batch_idx / len(data_loader), lossesdict['ce'], lossesdict['mse'], lossesdict['nce']))
    adjust_learning_rates(epoch)
    end_time = time.time()
    print("Time for epoch pass {}".format(end_time-start_time))
    train_loss = {'ce': float(np.mean(losses['ce'])), 'mse': float(np.mean(losses['mse'])), 'nce':float(np.mean(losses['nce']))}
    train_acc = correct / float(len(data_loader.dataset)*4)
    print('Train set: Average loss: CE {:.4f}, MSE {:.4f}, NT-Xent {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(train_loss['ce'], train_loss['mse'], train_loss['nce'], correct, len(data_loader.dataset)*4, 100.0*train_acc))
    return train_loss, train_acc

def validate(data_loader, epoch, log_interval=50):
    
    Networks['feature'].eval()
    Networks['classifier'].eval()
    Networks['transformation'].eval()
    
    losses = {'ce':[], 'mse':[], 'nce':[]}
    correct = 0
    valid_loss = np.Inf
    valid_acc = 0.0
    start_time = time.time()
    for batch_idx, sample in enumerate(data_loader): 
        with torch.no_grad():
            lossesdict = AlexNetDecoupling(batch_idx, sample, train=False)
        
        losses['ce'].append(lossesdict['ce'])
        losses['mse'].append(lossesdict['mse'])
        losses['nce'].append(lossesdict['nce'])
        correct += lossesdict['correct']
        if batch_idx % log_interval == 0:
            print('Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: CE {:.6f}, MSE {:.6f}, NT-Xent {:.6f}'.format(epoch, batch_idx*len(sample[0]), len(data_loader.dataset), 100. * batch_idx / len(data_loader), lossesdict['ce'], lossesdict['mse'], lossesdict['nce']))
    
    end_time = time.time()
    print("Time for epoch pass {}".format(end_time-start_time))
    valid_loss = {'ce': float(np.mean(losses['ce'])), 'mse': float(np.mean(losses['mse'])), 'nce':float(np.mean(losses['nce']))}
    valid_acc = correct / float(len(data_loader.dataset)*4)
    print('Valid set: Average loss: CE {:.4f}, MSE {:.4f}, NT-Xent {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(valid_loss['ce'], valid_loss['mse'], valid_loss['nce'], correct, len(data_loader.dataset)*4, 100.0*valid_acc))
    return valid_loss, valid_acc

def run_main_loop(loaders, num_epochs):
    writer = SummaryWriter('./logs/AlexNet_Unsupervised_Decoupling_Contrastive')
    save_path = "weights/AlexNet_Decoupling_Contrastive.pth"
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Performing {}th epoch".format(epoch))
        train_loss, train_acc = train(loaders['train_loader'], epoch)
        val_loss, val_acc = validate(loaders['valid_loader'], epoch)
        
        writer.add_scalar('CELoss/train', train_loss['ce'], epoch)
        writer.add_scalar('MSELoss/train', train_loss['mse'], epoch)
        writer.add_scalar('NT-XENTLoss/train', train_loss['nce'], epoch)
        
        writer.add_scalar('CELoss/Valid', val_loss['ce'], epoch)
        writer.add_scalar('MSELoss/Valid', val_loss['mse'], epoch)
        writer.add_scalar('NT-XENTLoss/Valid', val_loss['nce'], epoch)
        
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/Valid', val_acc, epoch)
    
        writer.add_scalar('LR', Optimizers['feature'].param_groups[0]['lr'], epoch)
        
        if val_acc > best_acc :
            best_acc = val_acc
            #save model
            states = {
                'epoch': epoch + 1,
                'feature_net':Networks['feature'].state_dict(),
                'classifier_net':Networks['classifier'].state_dict(),
                'transformation_net':Networks['transformation'].state_dict(),
                'feature_optimizer': Optimizers['feature'].state_dict(),
                'classifier_optimizer': Optimizers['classifier'].state_dict(),
                'transformation_optimizer': Optimizers['transformation'].state_dict(),
                'best_accuracy': best_acc
            }
            torch.save(states, save_path)
            print('Model Saved')


# In[9]:


run_main_loop(loaders, num_epochs)


# In[10]:


save_path = "weights/AlexNet_Decoupling_Contrastive200.pth"
states = {
                'feature_net':Networks['feature'].state_dict(),
                'classifier_net':Networks['classifier'].state_dict(),
                'transformation_net':Networks['transformation'].state_dict(),
                'feature_optimizer': Optimizers['feature'].state_dict(),
                'classifier_optimizer': Optimizers['classifier'].state_dict(),
                'transformation_optimizer': Optimizers['transformation'].state_dict(), 
            }
torch.save(states, save_path)


# In[ ]:




