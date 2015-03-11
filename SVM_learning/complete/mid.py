import Vcontutil, Vcont, os, subprocess, sys, random, mlpy, time, math
import pickle as pk
import numpy as np
from multiprocessing import Pool

video_list = []
video_names = []
crop = []
mid_numbers = 842

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

def extract_list(List, index_list):
	Temp = []
	for i in index_list:
		Temp.append(List[i])
	List = Temp
	return List
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
	#temp = text.split(' ')
        #ele = temp[0]
        #target = folder + ele + '/' + ele + '.avi'
	#target = ele + '.avi'
	target = text
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

def linear_SVM(fv, Label, C, w):
        svm = Vcontutil.linearSVM_T(fv, Label, C, w)
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

	ys = sorted(y)
	num = 0
	i = 1
	ap = 0
	while num < num_pos:
		if Label[np.where(y == ys[len(ys) - i])[0][0]] > 0:
			num += 1 
			ap += float(num) / (i * num_pos)
		i += 1
	acc_pos = 1 - float(error_pos) / num_pos
	acc_neg = 1 - float(error_neg) / (y.shape[0] - num_pos)
	acc = (acc_pos + acc_neg) / 2
	print 'average accuracy: ' + str(acc)
	print 'positive accuracy: ' + str(acc_pos)
	print 'negative accuracy: ' + str(acc_neg)
	print 'average precision: ' + str(ap)
	#acc = Vcontutil.linearSVM_P(fv, Label, svm)
	return acc, acc_pos, acc_neg, ap

def get_group(clu, count):
	group = map(int, clu)
	random.shuffle(group)
	n = len(group) / count
	r = len(group) % count
	stamp = 0
	clustering = []
	for i in range(count):
		if r != 0:
			clustering.append(group[stamp : stamp + n + 1])
			r -= 1
			stamp += n + 1
		else:
			clustering.append(group[stamp : stamp + n])
			stamp += n
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

def get_seed_name(txt):
	a = txt.find('all')
        name = '_' + txt[0 : a]
	return name

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
		name = get_seed_name(cluster[0])
		print '\n####### ' + name + ' #######'
		if os.path.isfile(cla_path + name + '/' + name + '_' + str(len(cluster)) + '_accuracy.npy'):
                	print 'done'
                	return

		count = 5
		if len(cluster) - 2 < 5:
			count = len(cluster) - 1
		clustering = get_group(cluster[1 : len(cluster)], count)
		clustering_neg = get_group(range(fv_hmdb.shape[0]), count)
		svm = []
		for i in range(count):
			clu = []
			clu_neg = []
			for j in range(count):
				if j == i and count > 1:
					continue
				clu += clustering[j]
				clu_neg += clustering_neg[j]
			print clu
			fv_pos = convert_index2fv(clu, fv_mid)
			label_pos = np.ones(fv_pos.shape[0])
			fv_neg = convert_index2fv(clu_neg, fv_hmdb)
			label_neg = np.zeros(fv_neg.shape[0])
			fv = Vcontutil.numpyVstack(fv_pos, fv_neg)
			label = Vcontutil.numpyHstack(label_pos, label_neg)
			w = {0 : 1, 1 : 1}
			svm.append(linear_SVM(fv, label, 100, w))

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

			svm[i].save_model(cla_path + name + '/' + name + '_' + str(len(cluster)) + '_' + str(i) + '.model')

		if count > 1:
			acc_avg = float(sum(accuracy[: , 3])) / accuracy.shape[0]
			index = np.where(accuracy == max(accuracy[: , 0]))[0][0]
		else:
			acc_avg = accuracy[0]
			index = 0
		print 'cross-validation mean average precision: ' + str(acc_avg)
		#svm[index].save_model(cla_path + name + '/' + name + '_' + str(len(cluster)) + '.model')
		np.save(cla_path + name + '/' + name + '_' + str(len(cluster)) + '_accuracy', accuracy)
	else:
		return
	print 'Time cost: ' + str(round(time.time() - sTime, 3)) + 'second'

def get_tfidf(tfidf_count, group, K):
	bow_temp = np.zeros(tfidf_count.shape[1])
	bow = np.zeros(tfidf_count.shape[1])
	for ele in group:
		temp = np.sort(tfidf_count[int(ele)])
		for i in range(K + 1):
			i += 1
			bow_temp[np.where(tfidf_count[int(ele)] == temp[len(temp) - i])[0][0]] += tfidf_count[int(ele)][np.where(tfidf_count[int(ele)] == temp[len(temp) - i])[0][0]]
		#bow += tfidf_count[int(ele)]

	temp = np.sort(bow_temp)
        for i in range(K + 1):
        	i += 1
               	bow[np.where(bow_temp == temp[len(temp) - i])[0][0]] += bow_temp[np.where(bow_temp == temp[len(temp) - i])[0][0]]
	bow *= idf
	bow /= np.dot(bow, bow) ** 0.5
	return bow

def compute_entropy(bow):
	entropy = 0
    	for i in np.where(bow != 0)[0]:
            	entropy -= bow[i] * math.log(bow[i])
	return entropy

def Compute_term_acc(com_a, com_b):
	correct = 0
	for ele in com_a:
		if ele in com_b:
			correct += 1
	return float(correct) / len(com_b)

def Load_mAP(groups):
	Fir = True
	term_num = 10
	for i in groups:
		num = len(i)
		name = get_seed_name(i[0])
		#temp = np.load(cla_path + name + '/' + name + '_' + str(num) + '_accuracy.npy')
		bow_g = get_tfidf(tfidf_count, i[1 : len(i)], term_num)
		term_acc = 0
		for j in i[1 : len(i)]:
			bow_v = get_tfidf(tfidf_count, [j], term_num)
			term_acc += Compute_term_acc(np.where(bow_v >= np.sort(bow_v)[np.where(np.sort(bow_v) != 0)[0][0]])[0], np.where(bow_g >= np.sort(bow_g)[np.where(np.sort(bow_g) != 0)[0][0]])[0])
			
		term_acc /= num
		print term_acc
		#en = compute_entropy(bow_g)
		data = Vcontutil.numpyHstack(name, num)
		#data = Vcontutil.numpyHstack(data, float(sum(temp[:, 0])) / temp.shape[0])
		data = Vcontutil.numpyHstack(data, term_acc)
		if Fir == True:
			D = data
			B = bow_g
			Fir = False
		else:
			D = Vcontutil.numpyVstack(D, data)
			B = Vcontutil.numpyVstack(B, bow_g)
	return D

def get_number_info_from_group_data(group_data, index = 0, one_d = False):
	if one_d == True:
		temp = group_data
	else:
		temp = group_data[: , index]
	data = np.zeros(len(temp))
	xx = 0
	for i in temp:
		data[xx] = float(i)
		xx += 1
	return data

def expand_group(groups, term_acc):
        term_num = 10
	size = []
        for k in range(len(groups)):
		i = list(groups[k])
                bow_g = get_tfidf(tfidf_count, i[1 : len(i)], term_num)
		entries = np.delete(range(mid_numbers) , get_number_info_from_group_data(i[1 : len(i)], 0, True))
                for j in entries:
                        bow_v = get_tfidf(tfidf_count, [j], term_num)
                        acc = Compute_term_acc(np.where(bow_v >= np.sort(bow_v)[np.where(np.sort(bow_v) != 0)[0][0]])[0], np.where(bow_g >= np.sort(bow_g)[np.where(np.sort(bow_g) != 0)[0][0]])[0])
			if acc > term_acc[k]*1.5:
				groups[k].append(str(j))
		#print i[1 : len(i)]
		#print groups[k]
		print len(i), len(groups[k])
		size.append([len(i), len(groups[k])])
	return groups, size
'''				
### mid video set extracting
## Load Cluster by seeds seletion
clu_path = '/home/Hao/Work/Cmts/cmt_clu3.txt'
clu_file = open(clu_path, 'r')
cluster_temp = clu_file.read()
cluster = cluster_temp.split('\n')

## Initial path setting
cla_path = '/home/Hao/Work/cla/'

## Get Cluster Group
clu_group = get_clu(cluster[0 : len(cluster) - 2], cla_path)
## Get tfidf count
tfidf_count = np.load('/home/Hao/Work/Cmts/tfidf_count.npy')
idf = np.load('/home/Hao/Work/Cmts/mid_idf.npy')

## Get group data [name, length, mP, entropy]
#group_data = np.load('/home/Hao/Work/group_data.npy')
clu_group = np.load('/home/Hao/Work/group_expand.npy')
remove = []
for i in range(len(clu_group)):
	if len(clu_group[i]) < 6:
		remove.append(i)
clu_group = np.delete(clu_group, remove)
D =  Load_mAP(clu_group)
#np.save('/home/Hao/Work/group_data', D)

## Expanse group
term_acc = get_number_info_from_group_data(group_data, 2)
term_acc_index = np.where(term_acc >= 0.35)[0]
clu_group = extract_list(clu_group, term_acc_index)
[clu_group, size] = expand_group(clu_group, term_acc)
np.save('/home/Hao/Work/group_expand', clu_group)
np.save('/home/Hao/Work/group_size', size)

# Sort term accuracy and length of group for plotting
ta = D[:, 2]
l = D[:, 1]
tas = np.sort(ta)
ls = np.zeros(len(l))
for i in range(len(tas)):
	ls[i] = l[np.where(ta == tas[i])[0][0]]
np.savez('/home/Hao/Work/plot', l = ls, ta = tas)
'''
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
'''
### Reduce hmdbdata dimension
fv_hmdb = np.load('/home/Hao/Work/hmdb_total_fv.npy')
saved = np.load('/home/Hao/Work/Cmts/hmdb_seeds.npy')
fv_hmdb = convert_index2fv(saved, fv_hmdb)
np.save('/home/Hao/Work/hmdb_half_fv', fv_hmdb)
sys.exit()
'''

### Linear SVM learning
## Get mid file sort
file_path = '/home/Hao/Work/viral_data/mid_cmts/bow/sel/selK5_T5.pkl'
sel_file = pk.load(open(file_path,'r'))
cmtz = get_cmtz(sel_file)

## Load Cluster by seeds seletion
clu_path = '/home/Hao/Work/Cmts/cmt_clu3.txt'
clu_file = open(clu_path, 'r')
cluster_temp = clu_file.read()
cluster = cluster_temp.split('\n')

## Initial path setting
traing_set_path = '/home/Hao/Work/traing_set/'
cla_path = '/home/Hao/Work/cla/'

## Get Cluster Group
clu_group = get_clu(cluster[0 : len(cluster) - 2], cla_path)
group_data = np.load('/home/Hao/Work/group_data.npy')
term_acc = get_number_info_from_group_data(group_data, 2)
term_acc = np.where(term_acc >= 0.35)[0]
#print term_acc
leng = get_number_info_from_group_data(group_data, 1)
xxx = extract_list(leng, term_acc)
xxx = np.array(xxx)
#print np.mean(xxx)
#print len(np.where(xxx>=0)[0])
temp = []
for i in term_acc:
	if leng[i] >= 5:
		temp.append(i)
term_acc = temp
#print term_acc

## compute coverage
clu_group = extract_list(clu_group, term_acc)
clu_group = np.load('/home/Hao/Work/group_expand.npy')
remove = []
for i in range(len(clu_group)):
        if len(clu_group[i]) < 6:
                remove.append(i)
clu_group = np.delete(clu_group, remove)

coverage = np.zeros(len(cmtz))
for i in clu_group:
	for j in range(len(i) - 1):
		j += 1
		coverage[int(i[j])] = 1
c = 0
for i in coverage:
	if i == 1:
		c += 1
#print float(c) / len(cmtz)

## Load prepared fv
fv_mid = np.load('/home/Hao/Work/mid_total_fv.npy')
fv_hmdb = np.load('/home/Hao/Work/hmdb_0.4_fv.npy')

#cross_validation(clu_group[1])
#sys.exit()
#p = Pool(2)
#p.map(cross_validation, clu_group[0 : 2000])

#f = open('/home/Hao/Work/mid_list.txt', 'r')
#video_list = get_video_list(f, video_list)
#fv = get_total_fv(video_list)
#linear_SVM()
#get_group()
#f_acc = open('/home/Hao/Work/acc.txt','w')
#p = Pool(4)
#p.map(cross_validation, clu_group)
#cross_validation(clu_group[0])
'''
## accuracy & coverage compute
cla_path = '/home/Hao/Work/cla/'
group_acc = []
group_pre = []
xx = 0
for i in clu_group:
	name = get_seed_name(i[0])
	if os.path.isfile(cla_path + name + '/' + name + '_' + str(len(i)) + '_accuracy.npy') == False:
		continue
	xx += 1
	temp = np.load(cla_path + name + '/' + name + '_' + str(len(i)) + '_accuracy.npy')
	group_acc.append(float(sum(temp[:, 0])) / len(temp))
	group_pre.append(float(sum(temp[:, 3])) / len(temp))
#print group_acc[1]
#sys.exit()
acc_sort = sorted(group_acc)
length = []
for i in range(xx):
	index_temp = group_acc.index(acc_sort[i])
	length.append(group_pre[index_temp])
	if i < 1:
		print acc_sort[i], len(clu_group[index_temp]), clu_group[index_temp]
ls = np.array(length)
ta = np.array(acc_sort) 
np.savez('/home/Hao/Work/plot', l = ls, ta = ta)
sys.exit()

index = []
index.append(group_acc.index(acc_sort[len(acc_sort) - 1]))
for i in range(len(clu_group) - 1):
	group1 = clu_group[index[len(index) - 1]]
	index_temp = group_acc.index(acc_sort[len(acc_sort) - i - 1])
	group2 = clu_group[index_temp]
	if group2 > group1:
		temp = group1
		group1 = group2
		group2 = temp
	cov = 0
	for j in range(len(group2) - 1):
		if group2[j + 1] in group1:
			cov += 1
	if (float(cov) / len(group1)) < 0.6 and group_acc[index_temp] > 0.7:
		index.append(index_temp)
		print index_temp
	if len(index) == 500:
		break
print len(index)
print group_acc[index[len(index) - 1]]

remove = range(len(clu_group))
remove = np.delete(remove, index)
clu_group = np.delete(clu_group, remove)

coverage = np.zeros(len(cmtz))
for i in clu_group:
        for j in range(len(i) - 1):
                j += 1
                coverage[int(i[j])] = 1
c = 0
for i in coverage:
        if i == 1:
                c += 1
print float(c) / len(cmtz)
'''
'''
### Dense Trajectory Feature Extrating
f = open('/home/hao/Desktop/raw_list.txt', 'r')
video_list = get_video_list(f, video_list)
output_dir = '/media/hao/My Book/raw_features/'
p = Pool(4)
p.map(mid_features, video_list)
'''
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
