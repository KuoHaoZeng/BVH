import os
import sys
import time
import numpy as np
import hifun
import vlfeat
import random
from yael import ynumpy
import scipy.io as sio
from numpy import linalg as LA
import subprocess
import mlpy

# Load Initial setting
[VectorDim,Ini_Index,PreSet,SubSample,K,listStart,listNum]=hifun.Load_Fisher_Initial()

# Load FileBook
SplitDir='/home/al-farabi/Desktop/testTrainMulti_7030_splits/'
[filebook,pages]=hifun.Load_FileBook(SplitDir,listStart,listNum)

tStart=time.time()
FirAll=True
for i in range(pages):
        if filebook[i][0]['Section']!=1:
                continue

        # Set Features Path
	if PreSet==1:
   		fvDir='/home/al-farabi/Desktop/ttsplit1/gmm_ns'+str(K)+'/fv/train/'+filebook[i][0]['Chapter']+'/'
	elif PreSet==2:
		fvDir='/home/al-farabi/Desktop/ttsplit1/gmm_ns'+str(K)+'/fv/testing/'+filebook[i][0]['Chapter']+'/'

	Dir=os.listdir(fvDir)
	LenDir=len(Dir)
	Num=0
	Fir=True
	while Num < LenDir:
  	     	fvTemp=np.load(fvDir+Dir[Num])

		if Fir==True:
        		fv=fvTemp.copy()
       			Fir=False
        	else:
        		fv=np.vstack((fv,fvTemp))

		Num+=1

	ClassTemp=np.zeros(LenDir,np.int32)+i

	if FirAll==True:
		fvAll=fv.copy()
		ClassAll=ClassTemp.copy()
		FirAll=False
	else:
		fvAll=np.vstack((fvAll,fv))
		ClassAll=np.hstack((ClassAll,ClassTemp))

	
tEnd=time.time()
print('\nFisher vector loaded done ... Running '+str(round(tEnd-tStart,3))+'(s)\n')



tStart=time.time()
if PreSet==1:
	svm = mlpy.LibLinear(solver_type='l2r_l2loss_svc', C=100)
	svm.learn(fvAll, ClassAll)
	if os.path.exists('/home/al-farabi/Desktop/ttsplit1/gmm_ns'+str(K)+'/result')==False:
		subprocess.call('mkdir /home/al-farabi/Desktop/ttsplit1/gmm_ns'+str(K)+'/result', shell=True)
	svm.save_model('/home/al-farabi/Desktop/ttsplit1/gmm_ns'+str(K)+'/result/linear.model')
	Label=svm.pred(fvAll)
	error=ClassAll.shape[0]
	for i in range(ClassAll.shape[0]):
		if Label[i]!=ClassAll[i]:
			error-=1

	print('Accuracy is '+str(round(100*float(error)/float(ClassAll.shape[0]),3))+'%')
	tEnd=time.time()
	print('\nSVM Traning of FV done ... Running '+str(round(tEnd-tStart,3))+'(s)\n')
elif PreSet==2:
	svm = mlpy.LibLinear(solver_type='l2r_l2loss_svc', C=100)
	svm = svm.load_model('/home/al-farabi/Desktop/ttsplit1/gmm_ns'+str(K)+'/result/linear.model')
	Label = svm.pred(fvAll)
	error=ClassAll.shape[0]
        for i in range(ClassAll.shape[0]):
                if Label[i]!=ClassAll[i]:
                        error-=1
	print('Accuracy is '+str(round(100*float(error)/float(ClassAll.shape[0]),3))+'%')
	'''
	f_acc=file('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(listStart/10)+'_RBF_acc.txt','w')
	f_acc.write(str(p_acc[0])+'\n')
	f_acc.close()
	'''
	tEnd=time.time()
	print('\nSVM Testing of FV done ... Running '+str(round(tEnd-tStart,3))+'(s)\n')




