#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch

from CustomDataset import ImageFolder, custom_collate_fn
from torch.utils.data import DataLoader as DataLoader
import torchvision.transforms as transforms


def get_transforms(datasetname='imagenet', train=True, simclr=False):
    transforms_list_train = []
    transforms_list_test = []
    transforms_list_normalize = []
    
    mean_pix = [0.485, 0.456, 0.406]
    std_pix = [0.229, 0.224, 0.225]
    transforms_list_normalize = [transforms.ToTensor(), transforms.Normalize(mean=mean_pix, std=std_pix)]
    if train is True:
        transforms_list_train = [
                     transforms.Resize(256), 
                     transforms.RandomCrop(224), 
                     transforms.RandomHorizontalFlip(p=0.5)
                    ]
        if simclr is True:

            transforms_list_train = [
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), 
                transforms.RandomGrayscale(p=0.2),
                transforms.Resize(224)
            ]
        return transforms_list_train + transforms_list_normalize
    else:
        transforms_list_test = [
                                transforms.Resize(224)
                                #transforms.CenterCrop(224)
                                ]
        return transforms_list_test + transforms_list_normalize




#load data for supervised or unsupervised task,
def get_dataloaders(dataset_name='imagenet', batch_size=1, num_workers=0, unsupervised=True, simclr=False):
    if dataset_name == 'imagenet':
        dataset_dir = './tiny-imagenet-200'
        dataset_train = ImageFolder(dataset_dir+'/train',  transforms.Compose(get_transforms(dataset_name, train=True, simclr=simclr)), unsupervised=unsupervised, simCLR=simclr)
        dataset_test = ImageFolder(dataset_dir+'/test',  transforms.Compose(get_transforms(dataset_name, train=False, simclr=simclr)), unsupervised=unsupervised, simCLR=simclr)
        dataset_val = ImageFolder(dataset_dir+'/valid',  transforms.Compose(get_transforms(dataset_name, train=False, simclr=simclr)), unsupervised=unsupervised, simCLR=simclr)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn if (unsupervised is True) and (simclr is False) else None)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn if (unsupervised is True) and (simclr is False) else None)
        valid_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn if (unsupervised is True) and (simclr is False) else None)
        return {'train_loader': train_loader, 'valid_loader':valid_loader, 'test_loader':test_loader}
    
    if dataset_name == 'places':
        dataset_dir = './places'
        
        dataset_train = ImageFolder(dataset_dir+'/train',  transforms.Compose(get_transforms(dataset_name, train=True, simclr=simclr)), unsupervised=unsupervised, simCLR=simclr)
        
        dataset_val = ImageFolder(dataset_dir+'/valid',  transforms.Compose(get_transforms(dataset_name, train=False, simclr=simclr)), unsupervised=unsupervised, simCLR=simclr)
        
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn if (unsupervised is True) and (simclr is False) else None)
       
        valid_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn if (unsupervised is True) and (simclr is False) else None)
        
        return {'train_loader': train_loader, 'valid_loader':valid_loader}


#load only 20% of data
def get_short_dataloaders(dataset_name='imagenet', batch_size=1, num_workers=0, unsupervised=True, simclr=False):
    
    if dataset_name == 'imagenet':
        dataset_dir = './tiny-imagenet-200'
        dataset_train = ImageFolder(dataset_dir+'/short_train',  transforms.Compose(get_transforms(dataset_name, train=True, simclr=simclr)), unsupervised=unsupervised, simCLR=simclr)
        dataset_test = ImageFolder(dataset_dir+'/short_test',  transforms.Compose(get_transforms(dataset_name, train=False, simclr=simclr)), unsupervised=unsupervised, simCLR=simclr)
        dataset_val = ImageFolder(dataset_dir+'/short_val',  transforms.Compose(get_transforms(dataset_name, train=False, simclr=simclr)), unsupervised=unsupervised, simCLR=simclr)
        
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn if (unsupervised is True) and (simclr is False) else None)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn if (unsupervised is True) and (simclr is False) else None)
        valid_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn if (unsupervised is True) and (simclr is False) else None)
        return {'train_loader': train_loader, 'valid_loader':valid_loader, 'test_loader':test_loader}
    
    if dataset_name == 'places':
        dataset_dir = './places'
        
        dataset_train = ImageFolder(dataset_dir+'/short_train',  transforms.Compose(get_transforms(dataset_name, train=True, simclr=simclr)), unsupervised=unsupervised, simCLR=simclr)
        
        dataset_val = ImageFolder(dataset_dir+'/short_val',  transforms.Compose(get_transforms(dataset_name, train=False, simclr=simclr)), unsupervised=unsupervised, simCLR=simclr)
        
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=custom_collate_fn if (unsupervised is True) and (simclr is False) else None)
       
        valid_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn if (unsupervised is True) and (simclr is False) else None)
        
        return {'train_loader': train_loader, 'valid_loader':valid_loader}


