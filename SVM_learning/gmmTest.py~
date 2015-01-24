import os
import sys
import time
import numpy as np
import hifun
import random
from yael import ynumpy
import scipy.io as sio
import subprocess

K=int(sys.argv[1])

Data=np.load('gmmTestingData.npy')
TotalSample=Data.shape[0]
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
gmm = ynumpy.gmm_learn(Data, K, 3, 60, 0, 10)

if os.path.exists('/home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(TotalSample))==False:
        subprocess.call('mkdir /home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(TotalSample), shell=True)
np.savez('/home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(TotalSample)+'/gmm',w=gmm[0],mu=gmm[1],std=gmm[2],pca=pca_transform,mean=mean)
