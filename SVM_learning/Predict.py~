import os
import sys
import time
import numpy as np
import hifun
import vlfeat
from svm import *
from svmutil import *

# Load Initial setting
[VectorDim,Ini_Index,PreSet,SubSample,Kstart,K,listStart,listNum]=hifun.Load_Initial()

ClassAll=np.empty([0],dtype=np.int32)
BoWAll=np.empty([0,4000],dtype=np.longdouble)
FirAll=True

for tt in range(4):

	if PreSet==1:
		VideoNum=70
		if tt ==0:
			ClassFile=open('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(tt)+'_Len12_Class.txt','r')
			BoWFile=open('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(tt)+'_Len12_BoW.txt','r')
			listNum=12
		else:
			ClassFile=open('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(tt)+'_Len13_Class.txt','r')
                        BoWFile=open('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(tt)+'_Len13_BoW.txt','r')
			listNum=13
	elif PreSet==2:
		VideoNum=30
		if tt ==0:
			ClassFile=open('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(tt)+'_Len12_Class_Testing.txt','r')
			BoWFile=open('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(tt)+'_Len12_BoW_Testing.txt','r')
			listNum=12
		else:
			ClassFile=open('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(tt)+'_Len13_Class_Testing.txt','r')
                        BoWFile=open('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(tt)+'_Len13_BoW_Testing.txt','r')
			listNum=13	

	lines=len(ClassFile.readlines())
	Class=np.empty([listNum*VideoNum],dtype=np.int32)
	ClassFile.seek(0,0)
	indexX=0
	for xx in range(lines):
		Temp=ClassFile.readline()
		WTemp=np.empty([1],dtype='S128')
		Fir=True
		for yy in Temp:
			if yy == ',':
				Class[indexX]=np.longdouble(WTemp)
				WTemp=np.empty([1],dtype='S128')
				Fir=True
				indexX+=1
			elif yy == ']':
				Class[indexX]=np.longdouble(WTemp)
				break
			elif yy != '[':
				if Fir==True:
					WTemp=yy
					Fir=False
				else:
					WTemp+=yy
			
			
	#for zz in range(listNum*VideoNum-1):	
	#	print Class[zz]
	#print Class.shape
	#sys.exit()
	ClassFile.close()
	
	
	lines=len(BoWFile.readlines())
	BoW=np.empty([lines,4000],dtype=np.longdouble)
	BoWFile.seek(0,0)
	indexY=0
	indexX=0
	for xx in range(lines):
		Temp=BoWFile.readline()
		WTemp=np.empty([1],dtype='S128')
		Fir=True
		for yy in Temp:
			if yy == ',':
				BoW[indexY][indexX]=np.longdouble(WTemp)
				WTemp=np.empty([1],dtype='S128')
				Fir=True
				indexX+=1
			elif yy == ']':
				BoW[indexY][indexX]=np.longdouble(WTemp)
				indexY+=1
				indexX=0
				break
			elif yy != '[':
				if Fir==True:
					WTemp=yy
					Fir=False
				else:
					WTemp+=yy
			
			
	#for zz in range(4000):	
	#	print BoW[lines-1][zz]
	#print BoW.shape
	#sys.exit()
	BoWFile.close()

	if FirAll==True:
		ClassAll=Class.copy()
		BoWAll=BoW
		FirAll=False
	elif FirAll==False:
		ClassAll=np.hstack((ClassAll,Class))
		BoWAll=np.vstack((BoWAll,BoW))
	
	print('List '+str(tt)+' has been loaded ......')


Y=np.array(ClassAll).tolist()
X=np.array(BoWAll).tolist()

f_BoW=file('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/BoW.txt','w')
uu=0
for yyy in Y:
	f_BoW.write('+'+str(yyy)+' ')
	nn=1
	xx=X[uu]
	for xxx in xx:
		f_BoW.write(str(nn)+':'+str(xxx)+' ')
		nn+=1	
	f_BoW.write('\n')
	print('Number '+str(uu)+' done.')
	uu+=1
f_BoW.close()

'''
prob = svm_problem(Y,X)
if PreSet==1:
	m = svm_train(prob, '-t 2 -c '+str(SubSample))
	svm_save_model('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(listStart/10)+'.model',m)
elif PreSet==2:
	m = svm_load_model('/home/al-farabi/Desktop/codebook_pool/hmdb51_org/list'+str(listStart/10)+'.model')
[p_labels, p_acc, p_vals] = svm_predict(Y,X,m)
'''

