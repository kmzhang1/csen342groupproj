from __future__ import print_function

import os
import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.stanford import get_stanford_dataloaders
from dataset.tinyimage import get_tiny_dataloaders

from helper.util import adjust_learning_rate, set_seed
from helper.loops import train_vanilla as train1, train_fer as train2, validate

from distiller_zoo import Softmax_T, KL

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=1600, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=240, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')  
    
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed') # in most experiments, seeds 1,5, 7 are used

    # dataset
    parser.add_argument('--model', type=str, default='ResNet18', choices=['wrn_40_2', 'MobileNetV2', 'ShuffleV2', 'ResNet18', 'Efficient'])

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'stanford', 'tiny'], help='dataset')
    
    parser.add_argument('-r', '--gamma', type=float, default=0.1, help='minimum weight for CE, typically set to 0.1')
    
    parser.add_argument('-i', '--init', type=int, default=1, help='initial epochs if needed, here simply set it to 1')

    parser.add_argument('--mu', type=float, default=0.9, help='mu')
     
    parser.add_argument('--kd_T', type=float, default=5, help='temperature for soft behavior')
    
    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')    
    

    opt = parser.parse_args()
    
    if opt.model in ['MobileNetV2', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/models'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.init == opt.epochs:
        opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}_std'.format(opt.model, opt.dataset, opt.learning_rate, opt.weight_decay, opt.trial)
    else:
        opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate, opt.weight_decay, opt.trial)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    
    best_acc = 0

    opt = parse_option()
    
    print(opt)
    
    
    if opt.seed is not None:
        set_seed(opt.seed)

    if opt.dataset == 'cifar100':
        
        print('cifar100')
                
        n_cls = 100
        get_data_loader = get_cifar100_dataloaders
    elif opt.dataset == 'stanford':
        get_data_loader = get_stanford_dataloaders
        n_cls = 120
    elif opt.dataset == 'tiny':
        get_data_loader = get_tiny_dataloaders
        n_cls = 500
    
    else:
        raise NotImplementedError(opt.dataset)

    # model
    if opt.model == 'Efficient':
        model =  torchvision.models.get_model('efficientnet_b0', num_classes=n_cls)
    else:
        model = model_dict[opt.model](num_classes=n_cls)

    
    
    model = model.cuda()
    print(opt.save_folder)
    if os.path.isfile(os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))):
        state = torch.load(os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model)))
    else:
        state = torch.load(os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model)))
    model.load_state_dict(state['model'])
    
    train_loader, val_loader, n_data = get_data_loader(batch_size=opt.batch_size, num_workers=opt.num_workers, 
                                                                is_instance=True,  is_shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    past = 0
    test_acc, test_acc_top5  = validate(val_loader, model, criterion, opt, past)

    #print(test_acc)
    #print(test_acc_top5)


if __name__ == '__main__':
    main()