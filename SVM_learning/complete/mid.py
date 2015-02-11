import Vcontutil, Vcont, os, subprocess, sys, random, mlpy, time
import pickle as pk
import numpy as np
from multiprocessing import Pool

#f = open('/home/al-farabi/Desktop/mid_list.txt', 'r')
#f = open('/home/al-farabi/Desktop/hmdb_list.txt', 'r')
video_list = []
video_names = []
crop = []

def get_video_list(f, video_list):
        for line in f:
                temp = line[0 : len(line) - 1]
	        video_list.append(temp)
	        #xx = temp.split('/')
	        #video_names.append(xx[len(xx) - 1])
        return video_list

'''
file_path = '/home/Hao/Work/viral_data/mid_cmts/bow/sel/selK5_T5.pkl'
sel_file = pk.load(open(file_path,'r'))
cmtz = []
for ele in sel_file:
	a = ele[2].find('all')
	b = ele[2].find('bow')
	cmtz.append('_' + ele[2][b + 4 : a])
'''
#folder = '/home/al-farabi/Desktop/mid/'
#dirs = os.listdir(folder)
#output_dir = '/home/al-farabi/Desktop/hmdb_features_fix360'
output_dir = '/home/al-farabi/Desktop/mid_features_fix360'

gmm_path = '/home/al-farabi/Desktop/fv/' # also a postive case path
'''
neg_tr_path = '/home/Hao/Work/nfv/train/'
neg_tr_list = os.listdir(neg_tr_path)
neg_tr_label = [0] * len(neg_tr_list)
neg_te_path = '/home/Hao/Work/nfv/testing/'
neg_te_list = os.listdir(neg_te_path)
neg_te_label = [0] * len(neg_te_list)

clu_path = '/home/Hao/Work/Cmts/cmt_clu2.txt'
clu_file = open(clu_path, 'r')
cluster_temp = clu_file.read()
cluster = cluster_temp.split('\n')

traing_set_path = '/home/Hao/Work/traing_set/'
cla_path = '/home/Hao/Work/cla/'
'''
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
        #target = folder + ele + '/' + ele + '.avi'
	target = ele + '.avi'
	Vcontutil.Extracting(target, output_dir)

def check_features_size(List, samp = 0):
        #Features = []
        size = 0
        for ele in List:
            #Feature = Vcontutil.Load_Unit_Features(output_dir + '/' + ele, samp)
            Feature = Vcontutil.Load_Unit_Features(ele, samp)
            size += Feature.shape[0]
            #Features = Vcontutil.numpyVstack(Features, Feature)
        return size

def get_features(List, samp):
        Features = []
        for ele in List:
            #Feature = Vcontutil.Load_Unit_Features(output_dir + '/' + ele, samp)
            Feature = Vcontutil.Load_Unit_Features(ele, samp)
            Features = Vcontutil.numpyVstack(Features, Feature)
        return Features

def mid_gmm(Features, sv_path, K = 256, nth = 1, nit = 30, redo = 1):
	# gmm training
	#Features = []
	gmm = 0
	if not (os.path.exists(sv_path + '/gmm.npz')):
        	#for ele in List:
                	#Feature = Vcontutil.Load_Unit_Features(output_dir + '/' + ele, samp)
                #	Feature = Vcontutil.Load_Unit_Features(ele, samp)
                #       Features = Vcontutil.numpyVstack(Features, Feature)
		#print
		#print Features.shape
		[gmm, pca_transform, mean] = Vcontutil.gmm_training(Features, K, nth, nit, redo)
	if gmm != 0:
		np.savez( sv_path + 'gmm', w = gmm[0], mu = gmm[1], std = gmm[2], pca = pca_transform, mean = mean)

def fisherGN(ele):
        temp = ele.split('/')
        name = temp[len(temp) - 1]
        if not os.path.exists(gmm_path + '/' + name + '.npy'):
                #gmm = Vcont.gmm_model(np.load(gmm_path + '/gmm.npz'))
                #Feature = Vcontutil.Load_Unit_Features(output_dir + '/' + ele, 0)
                Feature = Vcontutil.Load_Unit_Features(ele, 0)
                Vcontutil.fisher_vector(Feature, gmm, gmm_path + '/' + name)
        else:
                print ele + '.npy already exist!'

def linear_SVM(group, videoLabel, fv_neg, C = 100):
	fv=[]
	for n in group:
        	fvTemp = np.load(traing_set_path + n)
        	fv = Vcontutil.numpyVstack(fv, fvTemp)
		print n +' loading ...'
	fv = Vcontutil.numpyVstack(fv, fv_neg)
	Label = np.array(videoLabel)
        svm = Vcontutil.linearSVM_T(fv, Label, C)
        return svm
	#linearSVM_P(fv, Label)

def linear_pred(group, videoLabel, fv_neg, svm):
        fv=[]
        for n in group:
                fvTemp = np.load(traing_set_path + n)
                fv = Vcontutil.numpyVstack(fv, fvTemp)
                print n +' loading ...'
        fv = Vcontutil.numpyVstack(fv, fv_neg)
        Label = np.array(videoLabel)
        acc = Vcontutil.linearSVM_P(fv, Label, svm)
	return acc

def get_group(clu):
	group = clu[:]
	random.shuffle(group)
	n = len(group) / 5
	A = group[0 : n]
	B = group[n : 2 * n]	
	C = group[2 * n : 3 * n]
	D = group[3 * n : 4 * n]
	E = group[4 * n : len(group)]
	clustering = [A, B, C, D, E]
	return clustering

def convert_index2name(clu):
	name = []
	Label = []
	for ele in clu:
		name.append(cmtz[int(ele)] + '.npy')
		Label.append(1)
	return name, Label
'''
fv_tr_neg = np.load('/home/Hao/Work/neg_tr_fv.npy')
fv_te_neg = np.load('/home/Hao/Work/neg_te_fv.npy')

name = []
clu_group = []
for ele in cluster:
        temp = ele.split(' ')[0]
        if temp[len(temp) - 3 : len(temp)] == 'bow':
                a = temp.find('all')
                name.append('_' + temp[0 : a])
                if not os.path.exists(cla_path + name[len(name) - 1]):
                        subprocess.call('mkdir ' + cla_path + name[len(name) - 1], shell = True)
        clu_group.append(ele.split(' ')[0 : len(ele.split(' ')) - 1])

def cross_validation(cluster):
	#fv_tr_neg = []
        #fv_te_neg = []
	#for n in neg_te_list:
        #        fvTemp = np.load(traing_set_path + n)
        #        fv_te_neg = Vcontutil.numpyVstack(fv_te_neg, fvTemp)
	#	 print n + ' loading ......'
	#np.save('/home/Hao/Work/neg_te_fv', fv_te_neg)
	#sys.exit()
	#fv_tr_neg = np.load('/home/Hao/Work/neg_tr_fv.npy')
	#fv_te_neg = np.load('/home/Hao/Work/neg_te_fv.npy')
	sTime = time.time()
	if len(cluster) != 0:
		name = cluster[0]
		clustering = get_group(cluster[1 : len(cluster)])	
		accuracy = []
		svm = []
		for i in range(5):
			clu = []
			for j in range(5):
				if j == i:
					continue
				clu += clustering[j]
			[clu, clu_label] = convert_index2name(clu)
			clu_label += neg_tr_label
			svm.append(linear_SVM(clu, clu_label, fv_tr_neg))

			[clu, clu_label] = convert_index2name(clustering[i])
			clu_label += neg_te_label
			acc = linear_pred(clu, clu_label, fv_te_neg, svm[i])
			accuracy.append(acc)
		acc_avg = float(sum(accuracy)) / len(accuracy)
		print accuracy
		print 'cross-validation accuracy: ' + str(acc_avg)
		index = accuracy.index(max(accuracy))
		svm[index].save_model(cla_path + name + '/' + name + cluster[len(cluster) - 1] + '.model')
		#f.write(name + ': ' + str(acc_avg) + ' ' + str(max(accuracy)) + '\n')
	else:
		return
	print 'Time cost: ' + str(round(time.time() - sTime, 3)) + 'second'
'''

#linear_SVM()
#get_group()
#f_acc = open('/home/Hao/Work/acc.txt','w')
#p = Pool(4)
#p.map(cross_validation, clu_group)
#cross_validation(clu_group[0])

#p = Pool(4)
#p.map(mid_features, video_list)

#gmm = Vcont.gmm_model(np.load('/home/al-farabi/Desktop/fv/gmm.npz'))
f = open('/home/al-farabi/Desktop/hmdb_list.txt', 'r')
#gmm_path = '/home/al-farabi/Desktop/nfv/'
video_list = get_video_list(f, video_list)
#fisherGN(video_list[0])
Features = get_features(video_list, 38)
print Features.shape
#p = Pool(4)
#p.map(fisherGN, video_list)
f = open('/home/al-farabi/Desktop/mid_list.txt', 'r')
#gmm_path = '/home/al-farabi/Desktop/fv/'
#video_list = []
#video_list = get_video_list(f, video_list)
Features = Vcontutil.numpyVstack(Features, get_features(video_list, 913))
print Features.shape
mid_gmm(Features, gmm_path, 256, 4)
#fisherGN(video_names[243])
#p = Pool(4)
#p.map(fisherGN, video_list)
