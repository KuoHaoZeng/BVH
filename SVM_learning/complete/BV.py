from Vcont import *
from Vcontutil import *
from multiprocessing import Pool
import numpy as np
import sys
import os

print

### Load input list
List=Vlist(sys.argv[1])

### Load control parameter
options = []
i = 2
while i < len(sys.argv):
	options.append(sys.argv[i])
	i += 1
Control=Vcont_parameter(options)

### mutiple thread pre-setting
p=Pool(Control.nthread)

### Features Extracting
def featuresEX(n):
	if Control.features_force == True or os.path.exists(List.videoFolder[n] + List.videoName[n]) == False:	
		Extracting(List.videoPath[n])
print Control.nthread
print List.Len

p.map(featuresEX, List.Len)
sys.exit()
### Fisher Encoding
# gmm training
Features=[]
if Control.gmm_control == 1 or os.path.exists(os.getcwd() + '/gmm.npz') == False:
	for n in range(len(List.videoName)):
		Feature = Load_Unit_Features(List.videoFolder[n] + List.videoName[n], Control.gmm_subsample)
		Features = numpyVstack(Features, Feature)	
	gmm_training(Features, Control.gmm_K, Control.nthread, Control.gmm_nit, Control.gmm_redo)
# fisher vector generation
gmm=gmm_model(np.load('gmm.npz'))
def fisherGN(n):
	Feature = Load_Unit_Features(List.videoFolder[n] + List.videoName[n], 0)
	fisher_vector(Feature, gmm, List.videoFolder[n] + List.videoName[n])
p.map(fisherGN, List.Len)
