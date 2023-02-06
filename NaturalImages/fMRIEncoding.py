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

result_dir = 'result_64'
trainfeat = np.load('G:/SpikeCNNEncodingV8/NaturalImages/'+result_dir+'/trainfeat.npy')
testfeat = np.load('G:/SpikeCNNEncodingV8/NaturalImages/'+result_dir+'/testfeat.npy')

## subject1 ##
trainfmris1 = np.load('G:/SpikeCNNEncodingV8/NaturalImages/fmridata/dataTrnS1.npy')
testfmris1 = np.load('G:/SpikeCNNEncodingV8/NaturalImages/fmridata/dataValS1.npy')
roiS1 = np.load('G:/SpikeCNNEncodingV8/NaturalImages/fmridata/roiS1.npy') 

s1v1idx = np.where(roiS1==1)[0]
s1v2idx = np.where(roiS1==2)[0]
s1v3idx = np.where(roiS1==3)[0]
s1v3aidx = np.where(roiS1==4)[0]
s1v3bidx = np.where(roiS1==5)[0]
s1v4idx = np.where(roiS1==6)[0]


trainfmris1v1 = trainfmris1[s1v1idx,:]
testfmris1v1 = testfmris1[s1v1idx,:]
fmriencodings1v1 = fmriencoding(trainfeat, trainfmris1v1, testfeat, testfmris1v1, 3)
fmriencodings1v1.CalculateR2matrix(result_dir+'/r2mats1v1.pkl')
r2valuess1v1, replocs1v1 = fmriencodings1v1.SearchReploc(result_dir+'/r2mats1v1.pkl')
predfmris1v1, predaccs1v1 = fmriencodings1v1.PredictVoxActivity(replocs1v1)
# rvaluess1v1, tvaluess1v1, pvaluess1v1 = fmriencodings1v1.BootstrapProcedure(predfmris1v1)
# print(len(np.where(pvaluess1v1<0.01)[0]))
np.save(result_dir+'/predaccs1v1.npy',predaccs1v1)
np.save(result_dir+'/pvaluess1v1.npy',pvaluess1v1)
np.save(result_dir+'/predfmris1v1.npy',predfmris1v1)

trainfmris1v2 = trainfmris1[s1v2idx,:]
testfmris1v2 = testfmris1[s1v2idx,:]
fmriencodings1v2 = fmriencoding(trainfeat, trainfmris1v2, testfeat, testfmris1v2, 3)
fmriencodings1v2.CalculateR2matrix(result_dir+'/r2mats1v2.pkl')
r2valuess1v2, replocs1v2 = fmriencodings1v2.SearchReploc(result_dir+'/r2mats1v2.pkl')
predfmris1v2, predaccs1v2 = fmriencodings1v2.PredictVoxActivity(replocs1v2)
# rvaluess1v2, tvaluess1v2, pvaluess1v2 = fmriencodings1v2.BootstrapProcedure(predfmris1v2)
# print(len(np.where(pvaluess1v2<0.01)[0]))
np.save(result_dir+'/predaccs1v2.npy',predaccs1v2)
np.save(result_dir+'/pvaluess1v2.npy',pvaluess1v2)
np.save(result_dir+'/predfmris1v2.npy',predfmris1v2)

trainfmris1v3 = trainfmris1[s1v3idx,:]
testfmris1v3 = testfmris1[s1v3idx,:]
fmriencodings1v3 = fmriencoding(trainfeat, trainfmris1v3, testfeat, testfmris1v3, 3)
fmriencodings1v3.CalculateR2matrix(result_dir+'/r2mats1v3.pkl')
r2valuess1v3, replocs1v3 = fmriencodings1v3.SearchReploc(result_dir+'/r2mats1v3.pkl')
predfmris1v3, predaccs1v3 = fmriencodings1v3.PredictVoxActivity(replocs1v3)
# rvaluess1v3, tvaluess1v3, pvaluess1v3 = fmriencodings1v3.BootstrapProcedure(predfmris1v3)
# print(len(np.where(pvaluess1v3<0.01)[0]))
np.save(result_dir+'/predaccs1v3.npy',predaccs1v3)
np.save(result_dir+'/pvaluess1v3.npy',pvaluess1v3)
np.save(result_dir+'/predfmris1v3.npy',predfmris1v3)

voxel_num = 500
predfmri = np.concatenate((predfmris1v1, predfmris1v2, predfmris1v3), axis = 0)
truefmri = np.concatenate((testfmris1v1, testfmris1v2, testfmris1v3), axis = 0)
voxelidx = np.zeros([predfmri.shape[1],voxel_num])
for i in range(predfmri.shape[1]):
    pred = np.delete(predfmri, i, axis=1)
    true = np.delete(truefmri, i, axis=1)
    voxelcorr = np.zeros([predfmri.shape[0],1])
    for v in range(predfmri.shape[0]):
        corr, _ = pearsonr(pred[v,:],true[v,:])
        voxelcorr[v] = corr
    voxelcorr[np.isnan(voxelcorr)]=0
    voxelidx[i,:] = np.argsort(-voxelcorr,axis=0)[0:voxel_num].squeeze()
predfmri_ = np.zeros([voxel_num,predfmri.shape[1]])
for i in range(predfmri.shape[1]):
    predfmri_[:,i] = predfmri[np.int32(voxelidx[i,:]),i]
truefmri_ = np.zeros([voxel_num,truefmri.shape[1]])
for i in range(truefmri.shape[1]):
    truefmri_[:,i] = truefmri[np.int32(voxelidx[i,:]),i]

corrmats1 = np.zeros([truefmri.shape[1],truefmri.shape[1]])
for i in range(truefmri.shape[1]):
    for j in range(truefmri.shape[1]):
        corrmats1[i,j], _ = pearsonr(predfmri_[:,i],truefmri_[:,j])
    
  
idfcount = 0 
corrmax  = np.max(corrmats1, axis = 1)
corrmaxidx = np.argmax(corrmats1, axis = 1)
for i in range(truefmri.shape[1]):
    if corrmats1[i,i] == corrmax[i]:
        idfcount = idfcount + 1
idfaccs1 = idfcount/truefmri.shape[1]

fig = plt.figure()
sns_plot = sns.heatmap(corrmats1, cmap = 'YlGnBu_r') 
plt.title('correlation (r)')
image_list = range(0,121,20)
plt.xticks(image_list,image_list,rotation=0) 
plt.xlabel('Predicted fMRI activity (image number)')
plt.yticks(image_list,image_list,rotation=0) 
plt.ylabel('Measured fMRI activity (image number)')
plt.savefig('G:/SpikeCNNEncodingV8/manuscript/figures_new/fig.5/IdentificationPerformanceS1.jpg',dpi=600)
plt.show()

## subject2 ##
trainfmris2 = np.load('G:/SpikeCNNEncodingV8/NaturalImages/fmridata/dataTrnS2.npy')
testfmris2 = np.load('G:/SpikeCNNEncodingV8/NaturalImages/fmridata/dataValS2.npy')
roiS2 = np.load('G:/SpikeCNNEncodingV8/NaturalImages/fmridata/roiS2.npy') 

s2v1idx = np.where(roiS2==1)[0]
s2v2idx = np.where(roiS2==2)[0]
s2v3idx = np.where(roiS2==3)[0]
s2v3aidx = np.where(roiS2==4)[0]
s2v3bidx = np.where(roiS2==5)[0]
s2v4idx = np.where(roiS2==6)[0]


trainfmris2v1 = trainfmris2[s2v1idx,:]
testfmris2v1 = testfmris2[s2v1idx,:]
fmriencodings2v1 = fmriencoding(trainfeat, trainfmris2v1, testfeat, testfmris2v1, 3)
fmriencodings2v1.CalculateR2matrix(result_dir+'/r2mats2v1.pkl')
r2valuess2v1, replocs2v1 = fmriencodings2v1.SearchReploc(result_dir+'/r2mats2v1.pkl')
predfmris2v1, predaccs2v1 = fmriencodings2v1.PredictVoxActivity(replocs2v1)
# rvaluess2v1, tvaluess2v1, pvaluess2v1 = fmriencodings2v1.BootstrapProcedure(predfmris2v1)
# print(len(np.where(pvaluess2v1<0.01)[0]))
np.save(result_dir+'/predaccs2v1.npy',predaccs2v1)
np.save(result_dir+'/pvaluess2v1.npy',pvaluess2v1)
np.save(result_dir+'/predfmris2v1.npy',predfmris2v1)

trainfmris2v2 = trainfmris2[s2v2idx,:]
testfmris2v2 = testfmris2[s2v2idx,:]
fmriencodings2v2 = fmriencoding(trainfeat, trainfmris2v2, testfeat, testfmris2v2, 3)
fmriencodings2v2.CalculateR2matrix(result_dir+'/r2mats2v2.pkl')
r2valuess2v2, replocs2v2 = fmriencodings2v2.SearchReploc(result_dir+'/r2mats2v2.pkl')
predfmris2v2, predaccs2v2 = fmriencodings2v2.PredictVoxActivity(replocs2v2)
# rvaluess2v2, tvaluess2v2, pvaluess2v2 = fmriencodings2v2.BootstrapProcedure(predfmris2v2)
# print(len(np.where(pvaluess2v2<0.01)[0]))
np.save(result_dir+'/predaccs2v2.npy',predaccs2v2)
np.save(result_dir+'/pvaluess2v2.npy',pvaluess2v2)
np.save(result_dir+'/predfmris2v2.npy',predfmris2v2)

trainfmris2v3 = trainfmris2[s2v3idx,:]
testfmris2v3 = testfmris2[s2v3idx,:]
fmriencodings2v3 = fmriencoding(trainfeat, trainfmris2v3, testfeat, testfmris2v3, 3)
fmriencodings2v3.CalculateR2matrix(result_dir+'/r2mats2v3.pkl')
r2valuess2v3, replocs2v3 = fmriencodings2v3.SearchReploc(result_dir+'/r2mats2v3.pkl')
predfmris2v3, predaccs2v3 = fmriencodings2v3.PredictVoxActivity(replocs2v3)
# rvaluess2v3, tvaluess2v3, pvaluess2v3 = fmriencodings2v3.BootstrapProcedure(predfmris2v3)
# print(len(np.where(pvaluess2v3<0.01)[0]))
np.save(result_dir+'/predaccs2v3.npy',predaccs2v3)
np.save(result_dir+'/pvaluess2v3.npy',pvaluess2v3)
np.save(result_dir+'/predfmris2v3.npy',predfmris2v3)

voxel_num = 500
predfmri = np.concatenate((predfmris2v1, predfmris2v2, predfmris2v3), axis = 0)
truefmri = np.concatenate((testfmris2v1, testfmris2v2, testfmris2v3), axis = 0)
voxelidx = np.zeros([predfmri.shape[1],voxel_num])
for i in range(predfmri.shape[1]):
    pred = np.delete(predfmri, i, axis=1)
    true = np.delete(truefmri, i, axis=1)
    voxelcorr = np.zeros([predfmri.shape[0],1])
    for v in range(predfmri.shape[0]):
        if ~np.any(np.isnan(true[v,:])):
            corr, _ = pearsonr(pred[v,:],true[v,:])
            voxelcorr[v] = corr
    voxelcorr[np.isnan(voxelcorr)]=0
    voxelidx[i,:] = np.argsort(-voxelcorr,axis=0)[0:voxel_num].squeeze()
predfmri_ = np.zeros([voxel_num,predfmri.shape[1]])
for i in range(predfmri.shape[1]):
    predfmri_[:,i] = predfmri[np.int32(voxelidx[i,:]),i]
truefmri_ = np.zeros([voxel_num,truefmri.shape[1]])
for i in range(truefmri.shape[1]):
    truefmri_[:,i] = truefmri[np.int32(voxelidx[i,:]),i]

corrmats2 = np.zeros([truefmri.shape[1],truefmri.shape[1]])
for i in range(truefmri.shape[1]):
    for j in range(truefmri.shape[1]):
        corrmats2[i,j], _ = pearsonr(predfmri_[:,i],truefmri_[:,j])
    
  
idfcount = 0 
corrmax  = np.max(corrmats2, axis = 1)
corrmaxidx = np.argmax(corrmats2, axis = 1)
for i in range(truefmri.shape[1]):
    if corrmats2[i,i] == corrmax[i]:
        idfcount = idfcount + 1
idfaccs2 = idfcount/truefmri.shape[1]

fig = plt.figure()
sns_plot = sns.heatmap(corrmats2,cmap = 'YlGnBu_r') 
plt.title('correlation (r)')
image_list = range(0,121,20)
plt.xticks(image_list,image_list,rotation=0) 
plt.xlabel('Predicted fMRI activity (image number)')
plt.yticks(image_list,image_list,rotation=0) 
plt.ylabel('Measured fMRI activity (image number)')
plt.savefig('G:/SpikeCNNEncodingV8/manuscript/figures_new/fig.5/IdentificationPerformanceS2.jpg',dpi=600)
plt.show()


fig = plt.figure()
voxel_num = [0,100,500,1000,2000]
accs1 = [0,0.8833,0.9667,0.9667,0.8333]
accs2 = [0,0.7917,0.9083,0.7667,0.6417]
plt.plot(voxel_num,accs1,'ro-',voxel_num,accs2,'bo-')
plt.legend(['subject1','subject2'],loc = 'lower right')
plt.xlabel('number of voxels')
plt.ylabel('indentification accuracy')
plt.grid()
plt.savefig('figures/EncodingAccuracies.jpg',dpi=600)
plt.show()