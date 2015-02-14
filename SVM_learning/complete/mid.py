import Vcontutil, Vcont, os, subprocess, sys, random, mlpy, time
import pickle as pk
import numpy as np
from multiprocessing import Pool

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

def get_hmdb_list(video_list):
        label = []
        names = []
        for ele in video_list:
                label.append(int(ele.split(' ')[1]))
                temp = ele.split(' ')[0].split('/')
                names.append('/home/al-farabi/Desktop/fv_2_12/' + temp[len(temp) - 1].split('.')[0] + '.npy')
        label = np.array(label)
        return names, label

def get_cmtz(sel_file):
	cmtz = []
	for ele in sel_file:
		a = ele[2].find('all')
		b = ele[2].find('bow')
		cmtz.append('_' + ele[2][b + 4 : a])
	return cmtz

#folder = '/home/al-farabi/Desktop/mid/'
#dirs = os.listdir(folder)
#output_dir = '/home/al-farabi/Desktop/hmdb_features_fix360'
#output_dir = '/home/al-farabi/Desktop/mid_features_fix360'

#gmm_path = '/home/al-farabi/Desktop/fv/' # also a postive case path

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
        Feature = []
        for ele in List:
            #Feature = Vcontutil.Load_Unit_Features(output_dir + '/' + ele, samp)
            Feature = Vcontutil.Load_Unit_Features(ele, samp)
            Features = Vcontutil.numpyVstack(Features, Feature)
            Feature = []
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
                print name + ' go go !!'
                Feature = Vcontutil.Load_Unit_Features(ele, 0)
                Vcontutil.fisher_vector(Feature, gmm, gmm_path + '/' + name)
        else:
                print name + '.npy already exist!'

def linear_SVM(fv, Label, C = 100):
        svm = Vcontutil.linearSVM_T(fv, Label, C)
        return svm

def linear_pred(fv, Label, svm):
        w = svm.w()
	b = svm.bias()
	y = np.dot(fv, w) + b
	error_pos = 0
	num_pos = 0
	error_neg = 0
	for i in range(y.shape[0]):
		if Label[i] > 0:
			num_pos += 1
			if y[i] < 0:
				error_pos += 1
		else:
			if y[i] > 0:
				error_neg += 1
	acc_pos = 1 - float(error_pos) / num_pos
	acc_neg = 1 - float(error_neg) / (y.shape[0] - num_pos)
	acc = (acc_pos + acc_neg) / 2
	print 'average accuracy: ' + str(acc)
	print 'positive accuracy: ' + str(acc_pos)
	print 'negative accuracy: ' + str(acc_neg)
	#acc = Vcontutil.linearSVM_P(fv, Label, svm)
	return acc

def get_group(clu):
	group = map(int, clu)
	random.shuffle(group)
	n = len(group) / 5
	A = group[0 : n]
	B = group[n : 2 * n]	
	C = group[2 * n : 3 * n]
	D = group[3 * n : 4 * n]
	E = group[4 * n : len(group)]
	clustering = [A, B, C, D, E]
	return clustering

def convert_index2fv(clu, fv_all):
	temp = np.arange(fv_all.shape[0])
	temp = np.delete(temp, clu)
	fv = np.delete(fv_all, temp, 0)
	return fv

#fv_tr_neg = np.load('/home/Hao/Work/neg_tr_fv.npy')
#fv_te_neg = np.load('/home/Hao/Work/neg_te_fv.npy')

def get_clu(cluster, cla_path):
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
	return clu_group

def get_total_fv(List):
	fv = []
        for n in List:
                fvTemp = np.load(n)
                fv = Vcontutil.numpyVstack(fv, fvTemp)
                print n + ' loading ......'
        np.save('/home/al-farabi/Desktop/hmdb_dataset_fv_tes', fv)
	return fv

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
		a = cluster[0].find('all')
                name = '_' + cluster[0][0 : a]
		f = open(cla_path + name + '/' + name + str(len(cluster)) + '.txt', 'w')
		clustering = get_group(cluster[1 : len(cluster)])	
		clustering_neg = get_group(range(fv_hmdb.shape[0]))
		svm = []
		for i in range(5):
			clu = []
			clu_neg = []
			for j in range(5):
				if j == i:
					continue
				clu += clustering[j]
				clu_neg += clustering_neg[j]
			fv_pos = convert_index2fv(clu, fv_mid)
			label_pos = np.ones(fv_pos.shape[0])
			fv_neg = convert_index2fv(clu_neg, fv_hmdb)
			label_neg = np.zeros(fv_neg.shape[0])
			fv = Vcontutil.numpyVstack(fv_pos, fv_neg)
			label = Vcontutil.numpyHstack(label_pos, label_neg)
			svm.append(linear_SVM(fv, label))

                        fv_pos = convert_index2fv(clustering[i], fv_mid)
			label_pos = np.ones(fv_pos.shape[0])
                        fv_neg = convert_index2fv(clustering_neg[i], fv_hmdb)
                        label_neg = np.zeros(fv_neg.shape[0])
			fv = Vcontutil.numpyVstack(fv_pos, fv_neg)
                        label = Vcontutil.numpyHstack(label_pos, label_neg)
			acc = linear_pred(fv, label, svm[i])
			if i == 0:
				accuracy = np.array(acc)
			else:
				accuracy = Vcontutil.numpyVstack(accuracy, acc)
		acc_avg = float(sum(accuracy[:, 0])) / accuracy.shape[0]
		print accuracy
		print 'cross-validation accuracy: ' + str(acc_avg)
		index = np.where(accuracy == max(accuracy[:, 0]))[0][0]
		svm[index].save_model(cla_path + name + '/' + name + cluster[len(cluster) - 1] + '.model')
		f.write(name + ': ' + str(acc_avg) + ' ' + str(accuracy[index][0]) + ' ' + str(accuracy[index][1]) + ' ' + accuracy[index][2] + '\n')
		f.close()
	else:
		return
	print 'Time cost: ' + str(round(time.time() - sTime, 3)) + 'second'

'''
### Test for hmdb dataset
hmdb_list_path = '/home/al-farabi/Desktop/inList_tt1T.txt'
f = open(hmdb_list_path, 'r')
video_list = f.read()
video_list = video_list.split('\n')
[video_list, video_label] = get_hmdb_list(video_list[0 : len(video_list) - 1])
fv_tra = np.load('/home/al-farabi/Desktop/hmdb_dataset_fv_tra.npy')
svm = linear_SVM(fv_tra, video_label)

hmdb_list_path = '/home/al-farabi/Desktop/inList_tt1P.txt'
f = open(hmdb_list_path, 'r')
video_list = f.read()
video_list = video_list.split('\n')
[video_list, video_label] = get_hmdb_list(video_list[0 : len(video_list) - 1])
fv_tes = np.load('/home/al-farabi/Desktop/hmdb_dataset_fv_tes.npy')
acc = linear_pred(fv_tes, video_label, svm)
print acc
'''

### Linear SVM learning
## Get mid file sort
file_path = '/home/Hao/Work/viral_data/mid_cmts/bow/sel/selK5_T5.pkl'
sel_file = pk.load(open(file_path,'r'))
cmtz = get_cmtz(sel_file)

## Load Cluster by seeds seletion
clu_path = '/home/Hao/Work/Cmts/cmt_clu2.txt'
clu_file = open(clu_path, 'r')
a_path = '/home/Hao/Work/cla/'
cluster_temp = clu_file.read()
cluster = cluster_temp.split('\n')

## Initial path setting
traing_set_path = '/home/Hao/Work/traing_set/'
cla_path = '/home/Hao/Work/cla/'

## Get Cluster Group
clu_group = get_clu(cluster[770 : len(cluster) - 2], cla_path)

## Load prepared fv
fv_mid = np.load('/home/Hao/Work/mid_total_fv.npy')
fv_hmdb = np.load('/home/Hao/Work/hmdb_total_fv.npy')

#cross_validation(clu_group[0])
p = Pool(3)
p.map(cross_validation, clu_group)

#f = open('/home/Hao/Work/mid_list.txt', 'r')
#video_list = get_video_list(f, video_list)
#fv = get_total_fv(video_list)
#linear_SVM()
#get_group()
#f_acc = open('/home/Hao/Work/acc.txt','w')
#p = Pool(4)
#p.map(cross_validation, clu_group)
#cross_validation(clu_group[0])

### Dense Trajectory Feature Extrating
#p = Pool(4)
#p.map(mid_features, video_list)

'''
### Gmm model training
f = open('/home/al-farabi/Desktop/hmdb_list.txt', 'r')
video_list = get_video_list(f, video_list)
Features = get_features(video_list[:], 38)
print Features.shape
f = open('/home/al-farabi/Desktop/mid_list.txt', 'r')
video_list = []
video_list = get_video_list(f, video_list)
Features = Vcontutil.numpyVstack(Features, get_features(video_list[:], 913))
print Features.shape
mid_gmm(Features, gmm_path, 256, 4)
'''

'''
### Fisher Vector encoding
gmm = Vcont.gmm_model(np.load('/home/al-farabi/Desktop/fv/gmm.npz'))
f = open('/home/al-farabi/Desktop/hmdb_list.txt', 'r')
gmm_path = '/home/al-farabi/Desktop/nfv/'
video_list = get_video_list(f, video_list)
p = Pool(4)
p.map(fisherGN, video_list)

gmm = Vcont.gmm_model(np.load('/home/al-farabi/Desktop/fv/gmm.npz'))
f = open('/home/al-farabi/Desktop/mid_list.txt', 'r')
gmm_path = '/home/al-farabi/Desktop/nfv/'
video_list = []
video_list = get_video_list(f, video_list)
for qq in video_list:
    fisherGN(qq)
#p = Pool(4)
#p.map(fisherGN, video_list)
'''
