# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:01:11 2022

@author: 2
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
from utils import fmriencoding
import pickle
import os

result_dir = 'result'
scnnfeat = np.load('G:/SpikeCNNEncodingV8/HandwrittenCharacters/'+result_dir+'/stimulifeat.npy')
trainidx = loadmat('G:/DataSet/fMRI/Handwritten Characters/char_data/train_testset.mat')['trainidx']
testidx = loadmat('G:/DataSet/fMRI/Handwritten Characters/char_data/train_testset.mat')['testidx']

trainfeat = scnnfeat[trainidx-1,:,:,:].squeeze()
testfeat = scnnfeat[testidx-1,:,:,:].squeeze()

voxel_num = 500
output_dir = 'result/' + str(voxel_num) + 'voxels'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

## subject1 ##
fmris1v1= loadmat('G:/DataSet/fMRI/Handwritten Characters/char_data/XS01_V1.mat')['X']
fmris1v2= loadmat('G:/DataSet/fMRI/Handwritten Characters/char_data/XS01_V2.mat')['X_V2']

trainfmris1v1 = fmris1v1[trainidx-1,:].squeeze().T
testfmris1v1 = fmris1v1[testidx-1,:].squeeze().T
fmriencodings1v1 = fmriencoding(trainfeat, trainfmris1v1, testfeat, testfmris1v1, 3)
fmriencodings1v1.CalculateR2matrix(result_dir+'/r2mats1v1.pkl')
r2valuess1v1, replocs1v1 = fmriencodings1v1.SearchReploc(result_dir+'/r2mats1v1.pkl')
predfmris1v1, predaccs1v1 = fmriencodings1v1.PredictVoxActivity(replocs1v1)
# rvaluess1v1, tvaluess1v1, pvaluess1v1 = fmriencodings1v1.BootstrapProcedure(predfmris1v1)
# print(len(np.where(pvaluess1v1<0.01)[0]))

trainfmris1v2 = fmris1v2[trainidx-1,:].squeeze().T
testfmris1v2 = fmris1v2[testidx-1,:].squeeze().T
fmriencodings1v2 = fmriencoding(trainfeat, trainfmris1v2, testfeat, testfmris1v2, 3)
fmriencodings1v2.CalculateR2matrix(result_dir+'/r2mats1v2.pkl')
r2valuess1v2, replocs1v2 = fmriencodings1v2.SearchReploc(result_dir+'/r2mats1v2.pkl')
predfmris1v2, predaccs1v2 = fmriencodings1v2.PredictVoxActivity(replocs1v2)
# rvaluess1v2, tvaluess1v2, pvaluess1v2 = fmriencodings1v2.BootstrapProcedure(predfmris1v2)
# print(len(np.where(pvaluess1v2<0.01)[0]))
np.save(output_dir+'/predfmris1v1.npy',predfmris1v1)
np.save(output_dir+'/predfmris1v2.npy',predfmris1v2)

r2valuess1 = np.concatenate((r2valuess1v1, r2valuess1v2), axis = 0)
voxelidxs1 = np.argsort(-r2valuess1,axis=0)[0:voxel_num]  

replocs1 = np.concatenate((replocs1v1, replocs1v2), axis = 0)
trainfmris1 = np.concatenate((trainfmris1v1, trainfmris1v2), axis = 0)
testfmris1 = np.concatenate((testfmris1v1, testfmris1v2), axis = 0)
fmriencodings1 = fmriencoding(trainfeat, trainfmris1, testfeat, testfmris1, 3)
fmrivals1, predfmris1, predaccs1 = fmriencodings1.ImageReconstruction(voxelidxs1, replocs1)
np.save(output_dir+'/fmripreds1.npy',predfmris1)
np.save(output_dir+'/fmrivals1.npy',fmrivals1)
np.save(output_dir+'/predaccs1.npy',predaccs1)

## subject2 ##
fmris2v1= loadmat('G:/DataSet/fMRI/Handwritten Characters/char_data/XS02_V1.mat')['X']
fmris2v2= loadmat('G:/DataSet/fMRI/Handwritten Characters/char_data/XS02_V2.mat')['X_V2']

trainfmris2v1 = fmris2v1[trainidx-1,:].squeeze().T
testfmris2v1 = fmris2v1[testidx-1,:].squeeze().T
fmriencodings2v1 = fmriencoding(trainfeat, trainfmris2v1, testfeat, testfmris2v1, 3)
fmriencodings2v1.CalculateR2matrix(result_dir+'/r2mats2v1.pkl')
r2valuess2v1, replocs2v1 = fmriencodings2v1.SearchReploc(result_dir+'/r2mats2v1.pkl')
predfmris2v1, predaccs2v1 = fmriencodings2v1.PredictVoxActivity(replocs2v1)
# rvaluess2v1, tvaluess2v1, pvaluess2v1 = fmriencodings2v1.BootstrapProcedure(predfmris2v1)
# print(len(np.where(pvaluess2v1<0.01)[0]))

trainfmris2v2 = fmris2v2[trainidx-1,:].squeeze().T
testfmris2v2 = fmris2v2[testidx-1,:].squeeze().T
fmriencodings2v2 = fmriencoding(trainfeat, trainfmris2v2, testfeat, testfmris2v2, 3)
fmriencodings2v2.CalculateR2matrix(result_dir+'/r2mats2v2.pkl')
r2valuess2v2, replocs2v2 = fmriencodings2v2.SearchReploc(result_dir+'/r2mats2v2.pkl')
predfmris2v2, predaccs2v2 = fmriencodings2v2.PredictVoxActivity(replocs2v2)
# rvaluess2v2, tvaluess2v2, pvaluess2v2 = fmriencodings2v2.BootstrapProcedure(predfmris2v2)
# print(len(np.where(pvaluess2v2<0.01)[0]))

np.save(output_dir+'/predfmris2v1.npy',predfmris2v1)
np.save(output_dir+'/predfmris2v2.npy',predfmris2v2)
r2valuess2 = np.concatenate((r2valuess2v1, r2valuess2v2), axis = 0)
voxelidxs2 = np.argsort(-r2valuess2,axis=0)[0:voxel_num]  

replocs2 = np.concatenate((replocs2v1, replocs2v2), axis = 0)
trainfmris2 = np.concatenate((trainfmris2v1, trainfmris2v2), axis = 0)
testfmris2 = np.concatenate((testfmris2v1, testfmris2v2), axis = 0)
fmriencodings2 = fmriencoding(trainfeat, trainfmris2, testfeat, testfmris2, 3)
fmrivals2, predfmris2, predaccs2 = fmriencodings2.ImageReconstruction(voxelidxs2, replocs2)
np.save(output_dir+'/fmripreds2.npy',predfmris2)
np.save(output_dir+'/fmrivals2.npy',fmrivals2)
np.save(output_dir+'/predaccs2.npy',predaccs2)

## subject3 ##
fmris3v1= loadmat('G:/DataSet/fMRI/Handwritten Characters/char_data/XS03_V1.mat')['X']
fmris3v2= loadmat('G:/DataSet/fMRI/Handwritten Characters/char_data/XS03_V2.mat')['X_V2']

trainfmris3v1 = fmris3v1[trainidx-1,:].squeeze().T
testfmris3v1 = fmris3v1[testidx-1,:].squeeze().T
fmriencodings3v1 = fmriencoding(trainfeat, trainfmris3v1, testfeat, testfmris3v1, 3)
fmriencodings3v1.CalculateR2matrix(result_dir+'/r2mats3v1.pkl')
r2valuess3v1, replocs3v1 = fmriencodings3v1.SearchReploc(result_dir+'/r2mats3v1.pkl')
predfmris3v1, predaccs3v1 = fmriencodings3v1.PredictVoxActivity(replocs3v1)
# rvaluess3v1, tvaluess3v1, pvaluess3v1 = fmriencodings3v1.BootstrapProcedure(predfmris3v1)
# print(len(np.where(pvaluess3v1<0.01)[0]))

trainfmris3v2 = fmris3v2[trainidx-1,:].squeeze().T
testfmris3v2 = fmris3v2[testidx-1,:].squeeze().T
fmriencodings3v2 = fmriencoding(trainfeat, trainfmris3v2, testfeat, testfmris3v2, 3)
fmriencodings3v2.CalculateR2matrix(result_dir+'/r2mats3v2.pkl')
r2valuess3v2, replocs3v2 = fmriencodings3v2.SearchReploc(result_dir+'/r2mats3v2.pkl')
predfmris3v2, predaccs3v2 = fmriencodings3v2.PredictVoxActivity(replocs3v2)
# rvaluess3v2, tvaluess3v2, pvaluess3v2 = fmriencodings3v2.BootstrapProcedure(predfmris3v2)
# print(len(np.where(pvaluess3v2<0.01)[0]))

np.save(output_dir+'/predfmris3v1.npy',predfmris3v1)
np.save(output_dir+'/predfmris3v2.npy',predfmris3v2)
r2valuess3 = np.concatenate((r2valuess3v1, r2valuess3v2), axis = 0)
voxelidxs3 = np.argsort(-r2valuess3,axis=0)[0:voxel_num]  

replocs3 = np.concatenate((replocs3v1, replocs3v2), axis = 0)
trainfmris3 = np.concatenate((trainfmris3v1, trainfmris3v2), axis = 0)
testfmris3 = np.concatenate((testfmris3v1, testfmris3v2), axis = 0)
fmriencodings3 = fmriencoding(trainfeat, trainfmris3, testfeat, testfmris3, 3)
fmrivals3, predfmris3, predaccs3 = fmriencodings3.ImageReconstruction(voxelidxs3, replocs3)
np.save(output_dir+'/fmripreds3.npy',predfmris3)
np.save(output_dir+'/fmrivals3.npy',fmrivals3)
np.save(output_dir+'/predaccs3.npy',predaccs3)

