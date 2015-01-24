import os
import sys
import time
import numpy as np
import hifun
import vlfeat
from svm import *
from svmutil import *
from grid import *

# Load Initial setting
[VectorDim,Ini_Index,PreSet,SubSample,Kstart,K,listStart,listNum]=hifun.Load_Initial()

# Load FileBook
SplitDir='/home/al-farabi/Desktop/testTrainMulti_7030_splits/'
[filebook,pages]=hifun.Load_FileBook(SplitDir,listStart,listNum)

# Load Hicharchy Tree
TreeDir='/home/al-farabi/Desktop/codebook_pool/hmdb51_org/K'+str(K)+'.vlhkm'
tree = vlfeat._vlfeat.VlHIKMTree(0, 0)
tree=hifun.Load_Tree(TreeDir)
print('Loading codebook: K'+str(K)+'.vlhkm\n')

FirAll=True
ClassLab=0
for i in range(pages):
        if filebook[i][0]['Section']!=1:
                continue

        # Set Features Path
        featureDir='/home/al-farabi/Desktop/features_pool/hmdb51_org/'

        featureDir=featureDir+filebook[i][0]['Chapter']+'/'
	Dir=os.listdir(featureDir)
	Num=0
	LenDir=len(Dir)
	Fir=True
	while Num < LenDir:
		tStart=time.time()
        	[DataTemp,Num]=hifun.Load_Unit_Features(Num,featureDir,filebook[i],PreSet,SubSample)
		if Num==-1:
			break		

		DataTemp=DataTemp.T
		label=vlfeat.vl_hikmeanspush(tree,DataTemp,verb=1)
		DataTemp=0
		HistTemp=np.arange(K)
		HistTemp=np.float32(HistTemp-HistTemp)
		for xx in range(label.shape[1]):
			HistTemp[label[0,xx]-1]+=1
		Base=0
		for xx in range(K):
			Base+=(HistTemp[xx]**2)
		HistTemp/=(Base**0.5)
		
		if Fir==True:
                	Hist=HistTemp
                	Fir=False
        	else:
                	Hist=np.vstack((Hist,HistTemp))
		tEnd=time.time()
		print('Generate BoW of '+filebook[i][Num]['Filename']+' ... Running '+str(round(tEnd-tStart,3))+'(s)\n')
		Num+=1
	
	ClassTemp=np.zeros([Hist.shape[0]])+ClassLab+listStart

	if FirAll==True:
        	HistAll=Hist
		Class=ClassTemp
                FirAll=False
        else:
                HistAll=np.vstack((HistAll,Hist))
		Class=np.hstack((Class,ClassTemp))

	ClassLab+=1


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
'''
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

