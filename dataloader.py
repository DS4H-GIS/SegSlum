import os
import glob
import torch
import numpy as np
import pandas as pd
from skimage import io, transform
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as TF
import random
from PIL import Image
from torchvision import io
import copy

def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)
            
        img[y:y + erase_h, x:x + erase_w] = torch.from_numpy(value)
        mask[y:y + erase_h, x:x + erase_w] = 255

    return img, mask

class GPSDataset(Dataset):
    def __init__(self, metadata, root_dir, class_name, train, transform=None, cutout=False):
        self.metadata = pd.read_csv(metadata).values
        self.root_dir = root_dir
        self.transform = transform
        self.land_dir = os.path.join("./IND_Mumbai_z18/",class_name) 
        self.train = train
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata[idx].item())
        image_path = img_name + '.png'
        image = io.read_image(image_path)
        land_name = os.path.join(self.land_dir, self.metadata[idx].item())
        land_value =pd.read_csv(land_name + '.csv', encoding = "UTF-8", header = None)
        
        
        
        land_value = torch.tensor(land_value.values).unsqueeze(0)
        land_value = torch.clamp(land_value, max=1)
        
        if cutout:
            image, land_value = cutout(image, land_value, p=0.5)
        
         #VFlip
        if self.transform:
            image, land_value = self.transform(image, land_value)
            
  
        land_value = land_value.squeeze(0).long()

        return img_name, image, land_value
    
class GPSDataset2(Dataset):
    def __init__(self, metadata, root_dir, class_name, train, transform=None, s_transform=None, normalize=None):
        self.metadata = pd.read_csv(metadata).values
        self.root_dir = root_dir
        self.transform = transform
        self.s_transform = s_transform
        self.land_dir = os.path.join("./PAK_Karachi_z18/",class_name) 
        self.train = train
        self.normalize = normalize
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata[idx].item())
        image_path = img_name + '.png'
        image = io.read_image(image_path)
        land_name = os.path.join(self.land_dir, self.metadata[idx].item())
        try:
            land_value = Image.open(land_name + '.tif')
            land_value = torch.tensor(np.array(land_value)).unsqueeze(0)
            land_value = torch.clamp(land_value, max=1)
            
        except:
            land_value = torch.zeros(1,256,256)


        
         #VFlip
        if self.transform:
            image, land_value = self.transform(image, land_value)
            
        if self.s_transform != None:
            s_image, s_land_value = self.s_transform(image, land_value)
            s_image, s_land_value = cutout(s_image, s_land_value, p=0.5)
            s_image, s_land_value = self.normalize(s_image, s_land_value)
            s_land_value = s_land_value.squeeze(0).long()
        else:
            s_image, s_land_value = None, None
            
        image, land_value = self.normalize(image, land_value)
        
        land_value = land_value.squeeze(0).long()
        

        return img_name, image, land_value, s_image, s_land_value
    
    
class GPSDataset3(Dataset):
    def __init__(self, metadata, root_dir, class_name, train, transform=None, s_transform=None, normalize=None):
        self.metadata = pd.read_csv(metadata).values
        self.root_dir = root_dir
        self.transform = transform
        self.s_transform = s_transform
        self.land_dir = os.path.join("./PAK_Karachi_z18/",class_name) 
        self.train = train
        self.normalize = normalize
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata[idx].item())
        image_path = img_name + '.png'
        image = io.read_image(image_path)
        land_name = os.path.join(self.land_dir, self.metadata[idx].item())
        try:
            land_value = Image.open(land_name + '.tif')
            land_value = torch.tensor(np.array(land_value)).unsqueeze(0)
            land_value = torch.clamp(land_value, max=1)
            
        except:
            land_value = torch.zeros(1,256,256)
            
        if 1 in land_value:
            label_s = torch.tensor(1)
        else:
            label_s = torch.tensor(0)


        
         #VFlip
        if self.transform:
            image, land_value = self.transform(image, land_value)
            
        if self.s_transform != None:
            s_image, s_land_value = self.s_transform(image, land_value)
            s_image, s_land_value = cutout(s_image, s_land_value, p=0.5)
            s_image, s_land_value = self.normalize(s_image, s_land_value)
            s_land_value = s_land_value.squeeze(0).long()
        else:
            s_image, s_land_value = None, None
            
        image, land_value = self.normalize(image, land_value)
        
        land_value = land_value.squeeze(0).long()
        

        return img_name, image, land_value, s_image, s_land_value, label_s
    
class GPSDatasetT(Dataset):
    def __init__(self, metadata, root_dir, class_name, train, transform=None, cutout=False):
        self.metadata = pd.read_csv(metadata).values
        self.root_dir = root_dir
        self.transform = transform
        self.land_dir = os.path.join("./PAK_Karachi_z18/",class_name) 
        self.train = train
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata[idx].item())
        image_path = img_name + '.png'
        image = io.read_image(image_path)
        land_name = os.path.join(self.land_dir, self.metadata[idx].item())
        
        try:
            land_value = Image.open(land_name + '.tif')
            land_value = torch.tensor(np.array(land_value)).unsqueeze(0)
            land_value = torch.clamp(land_value, max=1)
            
        except:
            land_value = torch.zeros(1,256,256)
            
        if 1 in land_value:
            label_s = torch.tensor(1)
        else:
            label_s = torch.tensor(0)
        
        if cutout:
            image, land_value = cutout(image, land_value, p=0.5)
        
         #VFlip
        if self.transform:
            image, land_value = self.transform(image, land_value)
            
  
        land_value = land_value.squeeze(0).long()

        return img_name, image, land_value
    
class DistrictDataset(Dataset):
    def __init__(self, metadata, root_dir, transform=None):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        datalength = len(self.metadata)
        folder_idx = self.metadata.iloc[idx, 0]
       
        image_root_path = "{}{}".format(self.root_dir, folder_idx)
        images = np.stack([io.imread("{}/{}".format(image_root_path, x)) / 255.0 for x in os.listdir(image_root_path)])    
        sample = {'images': images, 'directory': folder_idx, 'num': len(images)}
        
        if self.transform:
            sample['images'] = self.transform(sample['images'])

        return sample  
    
    
class GPSDataset4(Dataset):
    def __init__(self, metadata, root_dir, class_name, train, transform=None, cutout=False):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir
        self.transform = transform
        self.land_dir = os.path.join("./ZAF_CapeTown_z18/",class_name) 
        self.train = train
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.metadata["image_id"][idx])
        image_path = img_name + '.png'
        image = io.read_image(image_path)
        land_name = os.path.join(self.land_dir, self.metadata["image_id"][idx])
        
        label_b = torch.tensor(self.metadata["answer"][idx].item())
        land_value = Image.open(land_name + '.tif')
        land_value = torch.tensor(np.array(land_value)).unsqueeze(0)
        land_value = torch.clamp(land_value, max=1)
        
        if 1 in land_value:
            label_s = torch.tensor(1)
        else:
            label_s = torch.tensor(0)
        
        if cutout:
            image, land_value = cutout(image, land_value, p=0.5)
        
         #VFlip
        if self.transform:
            image, land_value = self.transform(image, land_value)
            
  
        land_value = land_value.squeeze(0).long()

        return img_name, image, land_value, label_s, label_b
        
class ReducedDataset(Dataset):
    def __init__(self, metadata, root_dir):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        dir_list = self.metadata['Directory'].values.tolist()
        print(idx)
        if idx not in dir_list: 
            return -1
        feature_matrix = np.genfromtxt("{}{}.csv".format(self.root_dir, idx), delimiter=' ')                              
        sample = {'images': feature_matrix, 'directory': idx, 'num': len(feature_matrix)}
        return sample
    
class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, images):
        normalized = np.stack([F.normalize(x, self.mean, self.std, self.inplace) for x in images]) 
        return normalized
        
class ToTensor(object):
    def __call__(self, images):
        images = images.transpose((0, 3, 1, 2))
        return torch.from_numpy(images).float() 