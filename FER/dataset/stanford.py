from __future__ import print_function

import os
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

class stanford_dataset(Dataset):
    def __init__(self, root, transform):
        self.image_loader =  datasets.ImageFolder(root=root,
                                     transform=transform)

    def __len__(self):
        return len(self.image_loader)

    def __getitem__(self, index):
        image, target = self.image_loader[index]
        return image, target, index
        



def get_stanford_dataloaders(batch_size=128, num_workers=8, is_instance=False, is_shuffle=True):
    """
    cifar 100
    """
    
    train_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    

    train_set =stanford_dataset(root='dataset/train',
                                     transform=train_transform)
        
    test_set = stanford_dataset(root='dataset/test',
                                transform=test_transform)
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


