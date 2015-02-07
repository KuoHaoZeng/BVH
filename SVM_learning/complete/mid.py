import Vcontutil, os, subprocess, sys
import pickle as pk
import numpy as np
from multiprocessing import Pool

f=open('/home/Hao/Work/manual.txt','r')
video_list = []
inlist = []
crop = []
for line in f:
	temp = line[0 : len(line) - 1]
	video_list.append(temp)
	xx = temp.split(' ')
	inlist.append(xx[0])
	if len(xx)>3 and '*' in xx[3]:
		crop.append(xx[0])
'''
file_path = '/home/Hao/Work/selK5_T5.pkl'
sel_file = pk.load(open(file_path,'r'))
cmtz = []
for ele in sel_file:
	a = ele[2].find('all')
	b = ele[2].find('bow')
	cmtz.append('_' + ele[2][b + 4 : a])
'''
folder = '/home/Hao/Work/mid/'
#dirs = os.listdir(folder)
output_dir = '/home/Hao/Work/mid_features_360x240'

gmm_path = '/home/Hao/Work/fv/'

def ffmpeg_duration(target):
	FFMPEG_BIN = 'ffmpeg'
	command = [FFMPEG_BIN,'-i', target, '-']
	pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	pipe.stdout.readline()
	pipe.terminate()
	infos = pipe.stderr.read()
	index = infos.find('Duration:')
	duration = int(infos[index + 13 : index + 15]) * 60 + int(infos[index + 16 : index + 18])
	return duration

def ffmpeg_duration(target, output, start, end):
	subprocess.call('sudo ffmpeg -i ' + target + ' -ss ' + start + ' -t ' + end + ' -vcodec copy -acodec copy ' + output, shell = True)

def video_edit(text):
	temp = text.split(' ')
	ele = temp[0]
	target = folder + ele + '/' + ele + '.mp4'
        jukin = folder + ele + '/jukin' + ele + '.mp4'
	
	if os.path.isfile(jukin):
		return
	
	start = temp[1]
	end = temp[2]
	if not os.path.isfile(jukin):
		subprocess.call('sudo mv ' + target + ' ' + jukin, shell = True)
	
	ffmpeg_duration(jukin, target, start, end)
	print 'done!!'

def mid_features(text, f = None):
	temp = text.split(' ')
        ele = temp[0]
        target = folder + ele + '/' + ele + '.mp4'
	Vcontutil.Extracting(target, output_dir)

def mid_gmm(List, sv_path, samp = 10, K = 256, nth = 1, nit = 30, redo = 1):
	# gmm training
	Features=[]
	gmm = 0
	if not (os.path.exists(sv_path + '/gmm.npz')):
        	for ele in List:
                	Feature = Vcontutil.Load_Unit_Features(output_dir + '/' + ele, samp)
                	Features = Vcontutil.numpyVstack(Features, Feature)
		print
		print Features.shape
		[gmm, pca_transform, mean] = Vcontutil.gmm_training(Features, K, nth, nit, redo)
	if gmm != 0:
		np.savez( sv_path + 'gmm', w = gmm[0], mu = gmm[1], std = gmm[2], pca = pca_transform, mean = mean)


mid_gmm(inlist, gmm_path, 305, 256, 4)
#p = Pool(3)
#p.map(mid_features, video_list)
