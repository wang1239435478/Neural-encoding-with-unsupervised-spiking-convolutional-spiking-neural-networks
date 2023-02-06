# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:35:44 2022

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
import pickle
import os
from sklearn.utils import resample
from scipy import stats

class fmriencoding():
    def __init__(self, trainfeat, trainfmri, testfeat, testfmri, cv):
        self.trainfeat = trainfeat
        self.trainfmri = trainfmri
        self.testfeat = testfeat
        self.testfmri = testfmri
        self.cv = cv
        
    def CalculateR2matrix(self,savepath):
        fmri = self.trainfmri
        newmatrix = True
        if os.path.exists(savepath):
            with open(savepath,'rb') as f:
                r2mat = pickle.load(f)   
            endvoxel = r2mat['endvoxel']            
            startvoxel = endvoxel + 1
            newmatrix = False
            print('Data recover complete')                    
        else:
            r2mat = dict()
            startvoxel = 0
        with tqdm(total=fmri.shape[0]) as pbar:
            feat = self.trainfeat
            if newmatrix:
                r2matrix = np.zeros([fmri.shape[0],feat.shape[2],feat.shape[3],self.cv]) 
            else:
                r2matrix = r2mat['r2matrix']
                newmatrix = True
            for v in range(startvoxel,fmri.shape[0]):
                if ~np.any(np.isnan(fmri[v,:])):
                    for r in range(feat.shape[2]):
                        for c in range(feat.shape[3]):
                            reg = linear_model.LinearRegression()
                            r2matrix[v,r,c,:] = cross_val_score(reg, feat[:,:,r,c].squeeze(), fmri[v,:].squeeze(), cv=3, scoring='r2')
                pbar.update(1) 
                if (v+1)%20==0:                                                
                    r2mat['r2matrix'] = r2matrix
                    r2mat['endvoxel'] = v
                    with open(savepath,'wb') as f:
                        pickle.dump(r2mat,f,pickle.HIGHEST_PROTOCOL)
        r2mat['r2matrix'] = r2matrix
        r2mat['endvoxel'] = v
        with open(savepath,'wb') as f:
            pickle.dump(r2mat,f,pickle.HIGHEST_PROTOCOL)    
    
    def SearchReploc(self, r2matpath):
        with open(r2matpath,'rb') as f:
            r2mat = pickle.load(f)  
        r2values = np.zeros([self.trainfmri.shape[0],1])        
        reploc = np.zeros([self.trainfmri.shape[0],2])
        r2matrix = r2mat['r2matrix'] 
        r2matmean = np.mean(r2matrix, axis=-1)
        for v in range(self.trainfmri.shape[0]):
            r2matv = r2matmean[v,:,:]
            r2values[v] = np.max(r2matv)
            reploc[v,0] = np.where(r2matv == np.max(r2matv))[0][0]
            reploc[v,1] = np.where(r2matv == np.max(r2matv))[1][0]
        return r2values, reploc
    
    def PredictVoxActivity(self, reploc):
        predacc = np.zeros([self.trainfmri.shape[0],1])
        predfmri = np.zeros_like(self.testfmri)
        for v in range(self.trainfmri.shape[0]):
            reg = linear_model.LinearRegression()
            train_feat = self.trainfeat[:,:,np.int32(reploc[v,0]),np.int32(reploc[v,1])]
            test_feat = self.testfeat[:,:,np.int32(reploc[v,0]),np.int32(reploc[v,1])]
            if ~np.any(np.isnan(self.trainfmri[v,:])):
                reg.fit(train_feat, self.trainfmri[v,:].squeeze())
                predfmri[v,:] = reg.predict(test_feat)
            if ~np.any(np.isnan(self.testfmri[v,:])):
                predacc[v],_ = pearsonr(predfmri[v,:], self.testfmri[v,:]) 
        return predfmri, predacc
    
    def BootstrapProcedure(self, predfmri, n_samples = 100):
        rvalues = np.zeros([self.testfmri.shape[0],n_samples])
        for i in range(n_samples):
            bootstrappred, bootstraptrue = resample(predfmri.T,self.testfmri.T,n_samples=self.testfmri.shape[1],replace=True)
            for v in range(bootstrappred.shape[1]):     
                if ~np.any(np.isnan(self.testfmri[v,:])):
                    rvalues[v,i],_ = pearsonr(bootstrappred[:,v], bootstraptrue[:,v])

        tvalues = np.zeros([self.testfmri.shape[0],1])
        pvalues = np.zeros([self.testfmri.shape[0],1])        
        for v in range(bootstrappred.shape[1]):
            tvalues[v], pvalues[v] = stats.ttest_1samp(rvalues[v,:],0,alternative='greater')
        return rvalues,tvalues,pvalues
                
            
