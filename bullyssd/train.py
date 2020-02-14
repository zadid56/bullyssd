#########################################################
# This project is an implemtation of SSD for detecting the 
# cyberbullying from the image dataset.
# This project is done by Mhafuzul islam and Zadid Khan.
##########################################################
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import BullySSD
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from data.config import voc
import numpy as np
import argparse
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

print ('Imported OK')

#################################################
#              Global Params                    #
#################################################
dataset_root = '/scratch1/mdmhafi/'
weights_folder = 'weights/'
saved_folder = '/scratch1/mdmhafi/'
basenet = 'vgg.pth'
weight_decay = 5e-4
gamma = 0.1
learning_rate = 1e-3
batch_size = 32
momentum = 0.9
num_workers = 4
dataset_type = 'VOC'

voc_= {
    'num_classes': 3,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}
MEANS = (104, 117, 123)
#################################################
# Check the GPU settinngs
#################################################
if torch.cuda.is_available():
    print ('########### Using CUDA GPU! #############')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print ('########### No CUDA GPU found! #############')
    torch.set_default_tensor_type('torch.FloatTensor')
#################################################
# save the training loss into file.
#################################################
if not os.path.exists(weights_folder):
    os.mkdir(weights_folder)
f= open("train_loss.txt","w+")
f.close()
def save_to_file(value):
    f= open("train_loss.txt","a+")
    f.write('%.4f\n'%value)
    f.close()
###########################################
# Training 
##########################################
def train():
    cfg = voc
    ######################################
    # The SSD Model                      
    ######################################
    net = BullySSD('train',cfg['num_classes'])
    
    ######################################
    # Move the network to cuda
    ######################################
    cudnn.benchmark = True
    # Move the Net to CUDA #
    net = net.cuda()
    ######################################
    # Define the optimizer and loss functions
    ######################################
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum= momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, True)
    
    #####################################
    # Set the CUDA to Training mode. #
    #####################################
    net.train()
    
    #####################################
    # Load the dataset
    #####################################
    print('Loading the dataset...')
    dataset = VOCDetection(root= dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
    epoch_size = len(dataset) // batch_size
    data_loader = data.DataLoader(dataset, batch_size, num_workers=num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    print('Dataset Loaded!')
    #####################################
    # create batch iterator
    #####################################
    batch_iterator = iter(data_loader)
    
    #####################################
    # loss counters
    #####################################
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    train_loss = []
    
    print ('Total Epoch : ', cfg['max_iter'])
    
    ########################################
    #    Training statrts
    ########################################
    for iteration in range(cfg['max_iter']):
        # reset epoch loss counters
        loc_loss = 0
        conf_loss = 0
        epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, gamma, step_index)

        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        images = Variable(images.cuda())
        targets = [Variable(ann.cuda()) for ann in targets]
        
        ###############################
        # forward pass
        ###############################
        t0 = time.time()
        out = net(images)
        ###############################
        # backward back propagation
        ###############################
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data
        conf_loss += loss_c.data

        ####################################
        # saving intermediate training files
        ####################################
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data), end=' ')
            save_to_file(loss.data)
            
        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(),saved_folder + 'ssd300_VOC_' + repr(iteration) + '.pth')
    

    #######################################
    # save the trained weights
    ######################################
    save_path = save_folder + dataset_type + '.pth'
    torch.save(ssd_net.state_dict(), save_path)

##############################################
# some utiliy functions
##############################################
def adjust_learning_rate(optimizer, gamma, step):
    lr = learning_rate * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

#########################
#   Main function.
#########################
if __name__ == '__main__':
    train()

