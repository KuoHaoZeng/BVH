import os
import sys
import time
import numpy as np
import hifun
import vlfeat
from svm import *
from svmutil import *
from grid import *
import random
from yael import ynumpy
import scipy.io as sio
from numpy import linalg as LA
import subprocess
from multiprocessing import Pool

# Load Initial setting
[VectorDim,Ini_Index,PreSet,SubSample,K,listStart,listNum]=hifun.Load_Fisher_Initial()

# Load FileBook
SplitDir='/home/al-farabi/Desktop/testTrainMulti_7030_splits/'
[filebook,pages]=hifun.Load_FileBook(SplitDir,listStart,listNum)

# Load GMM model
gmmNP=np.load('/home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/gmm.npz')
gmm=[gmmNP['w'],gmmNP['mu'],gmmNP['std']]
pca_transform=gmmNP['pca']
mean=gmmNP['mean']

#tStart=time.time()
#for i in range(pages):
def fvGenerate(i):
        if filebook[i][0]['Section']!=1:
                #continue
		return 

        # Set Features Path
        featureDir='/home/al-farabi/Desktop/features_pool/hmdb51_org/'

        featureDir=featureDir+filebook[i][0]['Chapter']+'/'
	Dir=os.listdir(featureDir)
	LenDir=len(Dir)
	Num=0
	while Num < LenDir:
  	     	[DataTemp,Num]=hifun.Load_Unit_Features(Num,featureDir,filebook[i],PreSet,SubSample)
		if Num==-1:
			break		

		Data=0
		Data=DataTemp
		
		# apply the PCA to the image descriptor
   		Data = np.dot(Data - mean, pca_transform)
		fv_mu = ynumpy.fisher(gmm, Data, include = 'mu')
		fv_sigma = ynumpy.fisher(gmm, Data, include = 'sigma')

		fv=fv_mu.copy()
		fv=np.hstack((fv,fv_sigma))

		# power-normalization
		fv = np.sign(fv) * np.abs(fv) ** 0.5

		# L2 normalize
		norms = LA.norm(fv)
		fv /= norms
		
		if os.path.exists('/home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv')==False:
			subprocess.call('mkdir /home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv/', shell=True)

		if os.path.exists('/home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv/train')==False:
			subprocess.call('mkdir /home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv/train/', shell=True)

		if os.path.exists('/home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv/testing')==False:
			subprocess.call('mkdir /home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv/testing/', shell=True)

		if PreSet==1:
			if os.path.exists('/home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv/train/'+filebook[i][0]['Chapter'])==False:
				subprocess.call('mkdir /home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv/train/'+filebook[i][0]['Chapter'], shell=True)
	
			DirTemp=os.listdir(featureDir)
			indexNum=np.where(filebook[i]['Filename']==Dir[Num])
			np.save('/home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv/train/'+filebook[i][0]['Chapter']+'/'+filebook[i][indexNum[0][0]]['Filename'],fv)
		elif PreSet==2:
			if os.path.exists('/home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv/testing/'+filebook[i][0]['Chapter'])==False:
				subprocess.call('mkdir /home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv/testing/'+filebook[i][0]['Chapter'], shell=True)
	
			DirTemp=os.listdir(featureDir)
			indexNum=np.where(filebook[i]['Filename']==Dir[Num])
			np.save('/home/al-farabi/Desktop/ttsplit2/gmm_ns'+str(K)+'/fv/testing/'+filebook[i][0]['Chapter']+'/'+filebook[i][indexNum[0][0]]['Filename'],fv)
		Num+=1

tStart=time.time()	
nt=3
listpage=range(pages)
p=Pool(nt)
p.map(fvGenerate,listpage)
tEnd=time.time()
print('\nFisher encoding done ... Running '+str(round(tEnd-tStart,3))+'(s)\n')

'''
Y=np.array(Class).tolist()
X=np.array(HistAll).tolist()
f_BoW=file('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(listStart/10)+'_Len'+str(listNum)+'_BoW_Testing.txt','w')
f_Class=file('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(listStart/10)+'_Len'+str(listNum)+'_Class_Testing.txt','w')
for yyy in X:
	f_BoW.write(str(yyy)+'\n')
f_Class.write(str(Y)+'\n')
f_BoW.close()
f_Class.close()
#print(Y)
#print(X)
prob = svm_problem(Y,X)

tStart=time.time()
if PreSet==1:
	m = svm_train(prob, '-t 2 -c 10')
	svm_save_model('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(listStart/10)+'.model',m)
elif PreSet==2:
	#m = svm_load_model('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/K'+str(K)+'_Subsample'+str(SubSample)+'.model')
	m = svm_load_model('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(listStart/10)+'.model')
	[p_labels, p_acc, p_vals] = svm_predict(Y,X,m)
	f_acc=file('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(listStart/10)+'_RBF_acc.txt','w')
	f_acc.write(str(p_acc[0])+'\n')
	f_acc.close()
tEnd=time.time()
print('\nSVM Traning of BoV done ... Running '+str(round(tEnd-tStart,3))+'(s)\n')
'''

