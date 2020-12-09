#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import shutil
from random import shuffle


# In[2]:


##convert tiny-imagenet-200 dataset to ImageFolder Structure
#download & extract tiny-imagenet in tiny-imagenet-200
dataset_dir = './tiny-imagenet-200'
split_ratio = 0.8
dataset_train_target = dataset_dir+'/new_train'
dataset_test_tareget = dataset_dir+'/new_test'
dataset_valid_target = dataset_dir+'/new_valid'
dataset_train_source = dataset_dir+'/train'
dataset_val_source = dataset_dir+'/val'

try:
    os.mkdir(dataset_dir+'/new_train')
    os.mkdir(dataset_dir+'/new_test')
    os.mkdir(dataset_dir+'/new_valid')
except:
    pass


# In[ ]:


#convert validation images to pytorch ImageFOlder structure
with open(dataset_val_source+'/val_annotations.txt', "r") as f:
    for line in f:
        line = line.strip().split()
        image_name = line[0]
        target_folder = line[1]
        src_file = dataset_val_source+'/images/'+image_name
        dest_file = dataset_valid_target+'/'+target_folder+'/'+image_name
        try:
            os.mkdir(dataset_valid_target+'/'+target_folder)
        except:
            pass
        shutil.move(src_file, dest_file)


# In[3]:


#for all images per class split in train & test according to split ratio
import math
train_folders = glob.glob(dataset_train_source+'/*')

for folder in train_folders:
    target = folder.split('\\')[1]
    folder += '/images'
    images = glob.glob(folder+'/*')
    shuffle(images)
    split_n = math.floor(len(images)*split_ratio)
    #print(split_n)
    train_imgs = images[:split_n]
    test_imgs = images[split_n:]
    #print(len(train_imgs), len(test_imgs))
    try:
        os.mkdir(dataset_train_target+'/'+target)
        os.mkdir(dataset_test_tareget+'/'+target)
    except:
        pass
    
    for train_img in train_imgs:
        shutil.move(train_img, dataset_train_target+'/'+target+'/')
    for test_img in test_imgs:
        shutil.move(test_img, dataset_test_tareget+'/'+target+'/')
    


# In[ ]:




