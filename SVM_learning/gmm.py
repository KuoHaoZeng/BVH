import os
import sys
import time
import numpy as np
import hifun
import random
from yael import ynumpy
import scipy.io as sio
import subprocess

# Load Initial setting
[VectorDim,Ini_Index,PreSet,SubSample,K,listStart,listNum]=hifun.Load_Fisher_Initial()

# Load FileBook
SplitDir='/home/al-farabi/Desktop/testTrainMulti_7030_splits/'
[filebook,pages]=hifun.Load_FileBook(SplitDir,listStart,listNum)


Fir=True
#for i in range(pages):
tStart=time.time()
for i in range(pages):
        if filebook[i][0]['Section']!=2:
                continue


        # Set Features Path
        featureDir='/home/al-farabi/Desktop/features_pool/hmdb51_org/'

        featureDir=featureDir+filebook[i][0]['Chapter']+'/'
	Dir=os.listdir(featureDir)
	LenDir=len(Dir)
	Num=0
	DataInClass=0
	FirInClass=True
	while Num < LenDir:
  	     	[DataTemp,Num]=hifun.Load_Unit_Features(Num,featureDir,filebook[i],PreSet,SubSample)
		if Num==-1:
			break		

		if FirInClass==True:
        	        DataInClass=DataTemp
       		        FirInClass=False
        	else:
               		DataInClass=np.vstack((DataInClass,DataTemp))

		Num+=1

	if Fir==True:
        	Data=DataInClass.copy()
       		Fir=False
        else:
        	Data=np.vstack((Data,DataInClass))
TotalSample=Data.shape[0]
tEnd=time.time()
print('\nFeatures loaded done ... Running '+str(round(tEnd-tStart,3))+'(s)\n')

np.save('gmmTestingData',Data)
sys.exit()

# compute mean and covariance matrix for the PCA
mean = Data.mean(axis = 0)
Data = Data - mean
cov = np.dot(Data.T, Data)

# compute PCA matrix and keep only 64 dimensions
eigvals, eigvecs = np.linalg.eig(cov)
perm = eigvals.argsort()                   # sort by increasing eigenvalue
pca_transform = eigvecs[:, perm[213:426]]   # eigenvectors for the 64 last eigenvalues

# transform sample with PCA (note that numpy imposes line-vectors,
# so we right-multiply the vectors)
Data = np.dot(Data, pca_transform)

# train GMM
gmm = ynumpy.gmm_learn(Data, K,3,10)

if os.path.exists('/home/al-farabi/Desktop/ttsplit1/gmm_ns'+str(TotalSample))==False:
	subprocess.call('mkdir /home/al-farabi/Desktop/ttsplit1/gmm_ns'+str(TotalSample), shell=True)
np.savez('/home/al-farabi/Desktop/ttsplit1/gmm_ns'+str(TotalSample)+'/gmm',w=gmm[0],mu=gmm[1],std=gmm[2],pca=pca_transform,mean=mean)


