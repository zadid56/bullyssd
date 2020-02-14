#####################################
# Test the model outputs.
#####################################
print ('Bully Net Testing!')

import os
import sys
import argparse
import numpy as np
import cv2
from data.config import voc
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
################################
# Globals
################################
saved_weight_path = 'weights/bullynet.pth'
################################
# Chekc the GPU
################################
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
################################
# Input argument
################################
input_img = sys.argv[1]
if (input_img==None):
    print ('Input file path not given!')
###############################
# Import the ssd module_path
###############################
from ssd import BullySSD
net = BullySSD('test',voc['num_classes'])    # initialize SSD
net.load_weights(saved_weight_path)
#################################
# Read Image File
#################################
image = cv2.imread(input_img,cv2.IMREAD_COLOR)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#################################
# Preprocess the input
#################################
x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
x = torch.from_numpy(x).permute(2, 0, 1)
#################################
# Push the image to gpu
#################################
xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
#################################
# Predict 
#################################

y = net(xx)

#################################
# Save the output image
#################################
from data import VOC_CLASSES as labels
from matplotlib import pyplot as plt

top_k=10
plt.figure(figsize=(10,10))
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
plt.imshow(rgb_image)  # plot the image for matplotlib
currentAxis = plt.gca()

detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.6:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.5f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
        color = colors[i]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
        j+=1
        
#############################
# Save the image
#############################
print ('saving the image @: predict.jpg')
plt.savefig('predict.jpg')
#############################
