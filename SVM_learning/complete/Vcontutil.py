import sys, os, time, subprocess, random, mlpy
import numpy as np
from numpy import linalg as LA
from yael import ynumpy
from Vcont import *

def SuForC(Path):
                Re = Path.replace(';','\;')
                Re = Re.replace('(','\(')
                Re = Re.replace(')','\)')
                Re = Re.replace('&','\&')
                Re = Re.replace(' ','\ ')
		return Re

def DigName(Path, str1, str2):
		Temp = Path.split(str2)
		Temp = Temp[0].split(str1)
		return Temp[len(Temp) - 1]

def Extracting(fulPath, svPath):
	# check whether .avi is existing or not
	CPath = SuForC(fulPath)
	CsvPath = SuForC(svPath)
	video = DigName(fulPath, '/', '.')
	if os.path.exists(fulPath) == False:
		print('Error: ' + fulPath + ' does not exist.')
		sys.exit()
	# remove oldder verison
	if os.path.exists(svPath + video) == True:
		subprocess.call('rm ' + svPath + video, shell = True)
	# conduct
	print(video + ' Features Extracting ......')
	tStart=time.time()
	subprocess.call('./Video ' + CPath, shell = True)
	subprocess.call('./DenseTrackStab ' + CPath + ' ' + CsvPath, shell = True)
	tEnd=time.time()
	print('Cost ' + str(round(tEnd-tStart,3)) + ' sec.\n')

def Load_Unit_Features(fulPath, subsample):
	# open and load the binary feature file
	with open(fulPath, 'rb') as f:
                raw = np.fromfile(f, np.float32)
	f.close()
	# if frameNum > subsample > 0, subsampling conduct
	VectorDim=426
        FrameNum = len(raw) / VectorDim
	if subsample != 0 and FrameNum > subsample:
        	FrameNumVec = np.arange(FrameNum - 1)
              	RandomSap = random.sample(FrameNumVec, subsample)
                DataTemp = np.zeros(VectorDim * subsample, dtype = np.float32)
                xx = 0
        	for j in RandomSap:
        		DataTemp[xx * VectorDim : (xx + 1) * VectorDim] = raw[j * VectorDim : (j + 1) * VectorDim].copy()
                	xx += 1
        	raw = DataTemp.copy()
	Data = raw.copy()
	Data.shape = -1, VectorDim
		
	print('---Unit features loaded done---')
	print('Here is: ' + fulPath + '\n')

	return Data

def numpyVstack(vA, vB):
	if vA == []:
		vA = vB.copy()
	else:
		vA = np.vstack((vA, vB))

	return vA

def numpyHstack(vA, vB):
        if vA == []:
                vA = vB.copy()
        else:
                vA = np.hstack((vA, vB))

        return vA
		
def gmm_training(Data, K, nt=1, nit=10, redo=1):
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
	gmm = ynumpy.gmm_learn(Data, K, nt, nit, 0, redo)

	return gmm, pca_transform, mean
	#np.savez('gmm', w = gmm[0], mu = gmm[1], std = gmm[2], pca = pca_transform, mean = mean)

def fisher_vector(Data, gmm, fulPath): #gmm is a Gmm model class and can be seen in Vcont.py
	# apply the PCA to the image descriptor
   	Data = np.dot(Data - gmm.mean, gmm.pca)
	fv_mu = ynumpy.fisher(gmm.gmm, Data, include = 'mu')
	fv_sigma = ynumpy.fisher(gmm.gmm, Data, include = 'sigma')
	fv = fv_mu.copy()
	fv = np.hstack((fv, fv_sigma))
	# power-normalization
	fv = np.sign(fv) * np.abs(fv) ** 0.5
	# L2 normalize
	norms = LA.norm(fv)
	fv /= norms

	np.save(fulPath,fv)

def linearSVM_T(fvAll, ClassAll, c):
	svm = mlpy.LibLinear(solver_type='l2r_l2loss_svc', C=c)
        svm.learn(fvAll, ClassAll)
        #svm.save_model('linear.model')
        Label=svm.pred(fvAll)
        error=ClassAll.shape[0]
        for i in range(ClassAll.shape[0]):
                if Label[i]!=ClassAll[i]:
                        error-=1
        print('Learning accuracy is '+str(round(100*float(error)/float(ClassAll.shape[0]),3))+'%')
	return svm

def linearSVM_P(fvAll, ClassAll, svm):
	# check whether SVM model is existing or not
        #if os.path.exists('linear.model') == False:
        #        print('Error: SVM model does not exist.')
        #        sys.exit()
	#svm = svm.load_model('linear.model')
        Label = svm.pred(fvAll)
        error=ClassAll.shape[0]
        for i in range(ClassAll.shape[0]):
                if Label[i]!=ClassAll[i]:
                        error-=1
        print('Accuracy is '+str(round(100*float(error)/float(ClassAll.shape[0]),3))+'%\n')
	return round(100*float(error)/float(ClassAll.shape[0]),3)
