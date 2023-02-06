# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:36:19 2022

@author: 2
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms
import struct
import glob
import matplotlib.pyplot as plt

use_cuda = True
parser = argparse.ArgumentParser()
parser.add_argument("--max_epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=32)

parser.add_argument("--conv_kernel", type=int, default=5)
parser.add_argument("--conv_filters", type=int, default=64)
parser.add_argument("--conv_padding", type=int, default=2)
parser.add_argument("--conv_threshold", type=int, default=10)

parser.add_argument("--weight_mean", type=int, default=0.8) # controls the mean of the normal distribution used for initial random weights.
parser.add_argument("--weight_std", type=int, default=0.05) # controls the standard deviation of the normal distribution used for initial random weights.

parser.add_argument("--model_path", type=str, default="model")
parser.add_argument("--result_path", type=str, default="result")

args = parser.parse_args()
max_epochs = args.max_epochs
batch_size = args.batch_size

conv_kernel = args.conv_kernel
conv_filters = args.conv_filters
conv_padding = args.conv_padding
conv_threshold = args.conv_threshold
weight_mean = args.weight_mean
weight_std = args.weight_std
model_path = args.model_path
result_path = args.result_path 

if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(result_path):
    os.mkdir(result_path)


class SpikingCNN(nn.Module):
    def __init__(self, conv_filters, conv_padding, conv_kernel, conv_threshold, weight_mean, weight_std):    
        super(SpikingCNN, self).__init__()
        
        self.conv1 = snn.Convolution(6, conv_filters, conv_kernel, weight_mean, weight_std)
        self.conv1_t = conv_threshold
        self.k1 = 5
        self.r1 = 2
        self.conv_padding = conv_padding
        
        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))

        self.max_ap = Parameter(torch.Tensor([0.15]))
        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0

        
    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners
        
    def forward(self, input):
        input = sf.pad(sf.pooling(input.float(),2,2), (self.conv_padding,self.conv_padding,self.conv_padding,self.conv_padding), 0)
        if self.training:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)

            pot = sf.pointwise_inhibition(pot)
            spk = pot.sign()
            winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
            self.save_data(input, pot, spk, winners)
            return spk

        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, None, True)
            return spk, pot
  
    def stdp(self):
        self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])



def train_unsupervise(network, data):
	network.train()
	for i in range(len(data)):
		data_in = data[i]
		if use_cuda:
			data_in = data_in.cuda()
		network(data_in)
		network.stdp()

def test(network, data, target):
    network.eval()
    ans = [None] * len(data)
    t = [None] * len(data)
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        spk, pot = network(data_in)
        output = pot[-1,:,:,:]
        ans[i] = output.cpu().numpy()
        t[i] = target[i]
    return np.array(ans), np.array(t)

def compute_cl(network):
    network.train()
    weights = network.conv1.weight
    weights = weights*(1-weights)
    cl = torch.sum(weights, dim = (0, 1, 2, 3))/(conv_filters*6*conv_kernel*conv_kernel)
    return cl

class S1Transform:
    def __init__(self, filter, timesteps = 30):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.cnt = 0
        
    def __call__(self, image):
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()
    
kernels = [ utils.DoGKernel(3,3/9,6/9),
			utils.DoGKernel(3,6/9,3/9),
			utils.DoGKernel(7,7/9,14/9),
			utils.DoGKernel(7,14/9,7/9),
			utils.DoGKernel(13,13/9,26/9),
			utils.DoGKernel(13,26/9,13/9)]
filter = utils.Filter(kernels, padding = 6, thresholds = 50)

s1 = transforms.Compose([transforms.Grayscale(1),
    S1Transform(filter),])

data_root = 'G:/SpikeCNNEncodingV8/HandwrittenCharacters/TichValDataset'
dataset = utils.CacheDataset(ImageFolder(root = data_root, transform = s1))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

spikecnn = SpikingCNN(conv_filters, conv_padding, conv_kernel, conv_threshold, weight_mean, weight_std)
if use_cuda:
	spikecnn.cuda()

# Training The Model
print("Training the model")
cl_list = []
if os.path.isfile(model_path + "/saved_l1.net"):
	spikecnn.load_state_dict(torch.load(model_path + "/saved_l1.net"))
else:
    for epoch in range(max_epochs):
        print("Epoch", epoch)
        iter = 0
        for data,_ in data_loader:
            train_unsupervise(spikecnn, data)
            iter+=1
        cl = compute_cl(spikecnn).cpu().numpy()
        cl_list.append(cl)
        print('convergent value:', cl)
        if cl<0.01: break
    torch.save(spikecnn.state_dict(), model_path + "/saved_l1.net")

# Training The Second Layer
# print("Training the second layer")
# cl_list2 = []
# if os.path.isfile(model_path + "/saved_l2.net"):
# 	spikecnn.load_state_dict(torch.load(model_path + "/saved_l2.net"))
# else:
#     for epoch in range(max_epochs):
#         print("Epoch", epoch)
#         iter = 0
#         for data,_ in train_loader:
#             train_unsupervise(spikecnn, data, 2)
#             iter+=1
#         cl = compute_cl(spikecnn, 2).cpu().numpy()
#         cl_list2.append(cl)
#         print('convergent value:', cl)
#         if cl<0.01: break
#     torch.save(spikecnn.state_dict(), model_path + "/saved_l2.net")


# Extract stimuli feature
stimuli_root = "G:/SpikeCNNEncodingV4/fmri_dataset/stimuli" 
stimuli_dataset = utils.CacheDataset(ImageFolder(root = stimuli_root, transform = s1))
stimuli_loader = DataLoader(stimuli_dataset, batch_size=len(stimuli_dataset), shuffle=False)

for data,target in stimuli_loader:
    stimulifeat, _ = test(spikecnn, data, target)

np.save(result_path+'/stimulifeat.npy', stimulifeat)

tichval_root = 'TichValImages'
tichval_dataset = utils.CacheDataset(ImageFolder(root = tichval_root, transform = s1))
tichval_loader = DataLoader(tichval_dataset, batch_size=1, shuffle=False)
samplenum = 1
for data,target in tichval_loader:
    tichvalfeat, _ = test(spikecnn, data, target)
    np.save(result_path+'/tichvalfeat/tichvalfeatsample'+str(samplenum)+'.npy', tichvalfeat)
    samplenum = samplenum + 1