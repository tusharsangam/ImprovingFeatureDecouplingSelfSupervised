#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import glob


# In[2]:


try:
    os.mkdir("./tiny-imagenet-200/short_train")
    os.mkdir("./tiny-imagenet-200/short_test")
    os.mkdir("./tiny-imagenet-200/short_val")
except:
    pass


# In[5]:


target_folders = glob.glob("./tiny-imagenet-200/train/*")
for target_folder in target_folders:
    imgs = glob.glob(target_folder+'/*')
    imgs = imgs[:100]
    for img in imgs:
        dest_folder = target_folder.split("\\")[1]
        dest_file = "./tiny-imagenet-200/short_train/"+dest_folder
        try:
            os.mkdir(dest_file)
        except:
            pass
        shutil.copy(img, dest_file)


# In[6]:


target_folders = glob.glob("./tiny-imagenet-200/test/*")
for target_folder in target_folders:
    imgs = glob.glob(target_folder+'/*')
    imgs = imgs[:100]
    for img in imgs:
        dest_folder = target_folder.split("\\")[1]
        dest_file = "./tiny-imagenet-200/short_test/"+dest_folder
        try:
            os.mkdir(dest_file)
        except:
            pass
        shutil.copy(img, dest_file)


# In[7]:


target_folders = glob.glob("./tiny-imagenet-200/valid/*")
for target_folder in target_folders:
    imgs = glob.glob(target_folder+'/*')
    imgs = imgs[:100]
    for img in imgs:
        dest_folder = target_folder.split("\\")[1]
        dest_file = "./tiny-imagenet-200/short_val/"+dest_folder
        try:
            os.mkdir(dest_file)
        except:
            pass
        shutil.copy(img, dest_file)


# In[ ]:




