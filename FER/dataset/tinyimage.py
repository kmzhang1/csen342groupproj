from __future__ import print_function

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from datasets import load_dataset
import numpy as np

class tiny_dataset(Dataset):
    def __init__(self, split, transform):
        self.tiny = load_dataset('Maysee/tiny-imagenet', split=split)
        self.transform = transform
        
    def __len__(self):
        return len(self.tiny)

    def __getitem__(self, index):
        image = self.transform(self.tiny[index]['image'].convert('RGB'))/255.0
        target = self.tiny[index]['label']
        return image, target, index
        



def get_tiny_dataloaders(batch_size=128, num_workers=8, is_instance=False, is_shuffle=True):
    """
    cifar 100
    """
    
    train_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])

    

    train_set =tiny_dataset('train', train_transform)
        
    test_set = tiny_dataset('valid', test_transform)
    n_data = len(train_set)
            
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=is_shuffle,
                              num_workers=num_workers)
    


    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    
    else:
        return train_loader, test_loader


