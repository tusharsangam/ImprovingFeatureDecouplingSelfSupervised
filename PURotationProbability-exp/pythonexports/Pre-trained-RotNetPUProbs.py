#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ImageNet_RotNet_AlexNet.AlexNet import AlexNet as AlexNet
import torch
from DataLoader import get_dataloaders, get_short_dataloaders
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from torch import nn
from torch.nn import functional as F


# In[2]:


use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
nesterov = True
batch_size = 192
num_epochs = 35
LUT_lr = [(5, 0.01),(15, 0.002),(25, 0.0004),(35, 0.00008)]


# In[3]:


class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck, self).__init__()
        self.bottleneck = AlexNet(num_classes=4)
        pretrained_weights = "./ImageNet_RotNet_AlexNet/model_net_epoch50"
        pretrained_weights = torch.load(pretrained_weights)
        self.bottleneck.load_state_dict(pretrained_weights['network'])
        for param in self.bottleneck.parameters():
            param.requires_grad = False
    def forward(self, x):
        with torch.no_grad():
            x = self.bottleneck(x, ["fc_block"])
            #print(x.size())
        return x


# In[4]:


class RotNet(nn.Module):
    def __init__(self):
        super(RotNet, self).__init__()
        self.classifier = nn.Linear(4096, 1)
    def forward(self, x):
        x = self.classifier(x)
        return x


# In[5]:


bottleneck = BottleNeck().to(device)
rotnet = RotNet().to(device)


# In[6]:


optimizer = torch.optim.SGD(rotnet.parameters(), lr=learning_rate, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)


# In[7]:


criterion = torch.nn.BCEWithLogitsLoss()


# In[8]:


loaders = get_short_dataloaders('imagenet', batch_size=batch_size, num_workers=2)


# In[9]:


def adjust_lr(current_epoch):
    new_lr = next((lr for (max_epoch, lr) in LUT_lr if max_epoch>current_epoch), LUT_lr[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


# In[10]:


import time

def train(data_loader, model, epoch):
    model.train()
    losses = []
    correct = 0.0
    train_loss = np.Inf
    train_acc = 0.0
    #adjust_lr(epoch)
    start_time = time.time()
    for batch_idx, sample in enumerate(data_loader):
        
        optimizer.zero_grad()
        data, _ = sample
        batch_size = data.size(0)
        data =  data.to(device)
        data_90 = torch.flip(torch.transpose(data,2,3),[2])
        data_180 = torch.flip(torch.flip(data,[2]),[3])
        data_270 = torch.transpose(torch.flip(data,[2]),2,3)
        data = torch.stack([data, data_90, data_180, data_270], dim=1)
       
        batch_size, rotations, channels, height, width = data.size()
        data = data.view(batch_size*rotations, channels, height, width)
        
        #print(data.size())
        target = torch.FloatTensor([1]*batch_size+[0]*batch_size+[0]*batch_size+[0]*batch_size)
        #target.requires_grad = False
        target = target.to(device)
        
        randomize = np.arange(len(data))
        np.random.shuffle(randomize)
        #print(randomize)
        data = data[randomize]
        target = target[randomize]
        
        with torch.no_grad():
            feature = bottleneck(data)
        output = model(feature).squeeze_(dim=1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        pred = torch.sigmoid(output)
        zeros = torch.zeros_like(pred)
        ones = torch.ones_like(pred)
        pred = torch.where(pred>0.5, ones, zeros)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    end_time = time.time()
    print("Time for epoch pass {}".format(end_time-start_time))
    train_loss = float(np.mean(losses))
    train_acc = correct / float(len(data_loader.dataset)*4)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(train_loss, correct, len(data_loader.dataset)*4, 100*train_acc))
    return train_loss, train_acc


# In[11]:


def valid(data_loader, model):
    model.eval()
    losses = []
    correct = 0.0
    valid_loss = np.Inf
    valid_acc = 0.0
    start_time = time.time()
    for batch_idx, sample in enumerate(data_loader):
        with torch.no_grad():
            data, _ = sample
            batch_size = data.size(0)
            
            data = data.to(device)
            data_90 = torch.flip(torch.transpose(data,2,3),[2])
            data_180 = torch.flip(torch.flip(data,[2]),[3])
            data_270 = torch.transpose(torch.flip(data,[2]),2,3)
            data = torch.stack([data, data_90, data_180, data_270], dim=1)
            batch_size, rotations, channels, height, width = data.size()
            data = data.view(batch_size*rotations, channels, height, width)
            
            #print(data.size())
            target = torch.FloatTensor([1]*batch_size+[0]*batch_size+[0]*batch_size+[0]*batch_size)
            #target.requires_grad = False
            target = target.to(device)
            
            feature = bottleneck(data)
            output = model(feature).squeeze_(dim=1)
            loss = criterion(output, target)
            
            losses.append(loss.item())
             
            pred = torch.sigmoid(output)
            zeros = torch.zeros_like(pred)
            ones = torch.ones_like(pred)
            pred = torch.where(pred>0.5, ones, zeros)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    end_time = time.time()
    print("Time for valid epoch pass {}".format(end_time-start_time))    
    valid_loss = float(np.mean(losses))
    valid_acc = correct / float(len(data_loader.dataset)*4)
    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(valid_loss, correct, len(data_loader.dataset)*4,100*valid_acc))
    return valid_loss, valid_acc


# In[12]:


def run_main_loop(model, loaders, num_epochs):
    writer = SummaryWriter('./logs/AlexNet_PULearning')
    #os.unlink('./logs/AlexNet_PULearning')
    save_path = "weights/AlexNet_RotNet_PU.pth"
    best_acc = 0.0
    for epoch in range(num_epochs):
        print("Performing {}th epoch".format(epoch))
        train_loss, train_acc = train(loaders['train_loader'], model, epoch)
        val_loss, val_acc = valid(loaders['valid_loader'], model)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/Valid', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/Valid', val_acc, epoch)        
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        if val_acc > best_acc :
            best_acc = val_acc
            #save model
            states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_accuracy': best_acc
            }
            torch.save(states, save_path)
            print('Model Saved')


# In[13]:


run_main_loop(rotnet, loaders, num_epochs)


# In[ ]:


'''
import time
data_loader = loaders['valid_loader']
criterion = nn.BCEWithLogitsLoss()
correct = 0
train_loss = np.Inf
train_acc = 0.0
with torch.no_grad():
    start_time = time.time()
    losses = []
    for sample in data_loader:
        data, _ = sample
        batch_size = data.size(0)
        data =  data.to(device)
        data_90 = torch.flip(torch.transpose(data,2,3),[2])
        data_180 = torch.flip(torch.flip(data,[2]),[3])
        data_270 = torch.transpose(torch.flip(data,[2]),2,3)
        data = torch.stack([data, data_90, data_180, data_270], dim=1)
        batch_size, rotations, channels, height, width = data.size()
        data = data.view(batch_size*rotations, channels, height, width)
        
        target = torch.FloatTensor([0]*batch_size+[1]*batch_size+[1]*batch_size+[1]*batch_size)
        target = target.to(device)
        with torch.no_grad():
            feature = bottleneck(data)
        output = rotnet(feature)
        output = output.squeeze_(dim=1)
        
        loss = criterion(output, target)
        losses.append(loss.item())
        #output = F.softmax(output, dim=1)
        pred = torch.sigmoid(output) #.argmax(dim=1, keepdim=True)
        ones = torch.ones_like(pred)
        zeros = torch.zeros_like(pred)
        pred = torch.where(pred>0.5, ones, zeros)
        correct += pred.eq(target.view_as(pred)).sum().item()
        #print(correct)
    
    end_time = time.time()
    print("Time for epoch pass {}".format(end_time-start_time))
    train_loss = float(np.mean(losses))
    train_acc = correct / float(len(data_loader.dataset)*4)
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(train_loss, correct, len(data_loader.dataset)*4, 100*train_acc))


# In[ ]:




