import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc
from multibox import MultiBox
from l2norm import L2Norm2d
import os

class BullySSD(nn.Module):
   
    input_size = 300
    
    def __init__(self,phase,num_classes):
        super(BullySSD, self).__init__()
        ##################################
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward())
        self.size = 300
        ##################################
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)
        ##################################
        #define the Base Model First.
        self.base = self.VGG16()
        # This is the First Conf and Loc layer.
        self.norm4 = L2Norm(512, 20) #(20)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)
        
        self.multibox = MultiBox()
        
    def forward(self,x):
        
        out = self.base(x)
        
        hs = []
        hs.append(self.norm4(out))
        
        # VGG last 512 layers
        out = F.max_pool2d(out, kernel_size=2, stride=2, ceil_mode=True)
        
        # [512, 512, 512, M]
        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = F.max_pool2d(out, kernel_size=3, padding=1, stride=1, ceil_mode=True)
        
        # Branch #1
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        hs.append(out)  # conv7
        
        # Branch #2
        out = F.relu(self.conv8_1(out))
        out = F.relu(self.conv8_2(out))
        hs.append(out)  # conv8_2
        
        # Branch #3
        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        hs.append(out)  # conv9_2
        
        # Branch #4
        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        hs.append(out)  # conv10_2
        
        # Branch #5
        out = F.relu(self.conv11_1(out))
        out = F.relu(self.conv11_2(out))
        hs.append(out)  # conv11_2
        
        
        loc, conf = self.multibox(hs)
        
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        elif self.phase == 'train':
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
          

        return output #loc_preds, conf_preds
       
    ########################################
    # VGG 16 Netowrk
    ########################################
    def VGG16(self):
        '''VGG16 layers.'''
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)
    
    ###########################################
    # Load Weights
    # This function is for Testing purpose.
    ###########################################
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished Loading!')
        else:
            print('Sorry only .pth and .pkl files supported.')
    
