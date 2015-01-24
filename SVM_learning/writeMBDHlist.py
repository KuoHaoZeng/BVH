import os
import sys
import time
import numpy as np
import hifun
import random
from yael import ynumpy
import scipy.io as sio
import subprocess

# Load FileBook
SplitDir='/home/al-farabi/Desktop/testTrainMulti_7030_splits/'
[filebook,pages]=hifun.Load_FileBook(SplitDir,0,51)

#f=file('/home/al-farabi/Desktop/inList_tt1.txt','w')
for i in range(pages):
	if filebook[i][0]['Section']!=1:
                continue
	
	#f.write('/home/al-farabi/Desktop/video_pool/hmbd51_org/'+filebook[i][0]['Chapter']+'/'+filebook[i][0]['Filename']+' '+str(filebook[i][0]['Class'])+'\n')

#f.close()
