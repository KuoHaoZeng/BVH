import Vcontutil, Vcont, os, subprocess, sys, random, mlpy, time, math, sklearn.metrics, heapq, json
import pickle as pk
import numpy as np
from multiprocessing import Pool

video_list = []
video_names = []
crop = []
mid_numbers = 842
gmm = Vcont.gmm_model(np.load('/home/Hao/Work/gmm/gmm.npz'))
fv_path = '/media/Hao/My Book/raw_whole_fv_demo/'

def get_tuple_index(arr):
	index = np.zeros(len(arr))
	for i in xrange(len(arr)):
		index[i] = arr[i][0]
	return index

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

def check_features_size(List, samp = 0, raw = 0):
        #Features = []
        size = []
        for ele in List:
            #Feature = Vcontutil.Load_Unit_Features(output_dir + '/' + ele, samp)
		if raw == 0:
            		Feature = Vcontutil.Load_Unit_Features(ele, samp)
		else:
			[Feature, time_stamp] = Vcontutil.Load_Raw_Features(ele, samp)
		size.append(Feature.shape[0])
            #Features = Vcontutil.numpyVstack(Features, Feature)
        size = np.array(size)
	return size

def get_features(List, samp, raw = 0):
        Features = []
        Feature = []
	Stamp = []
	print len(Stamp)
        for ele in List:
            #Feature = Vcontutil.Load_Unit_Features(output_dir + '/' + ele, samp)
		if raw == 0:
        		Feature = Vcontutil.Load_Unit_Features(ele, samp)
		else:
			[Feature, time_stamp] = Vcontutil.Load_Raw_Features(ele, samp)
			Stamp = Vcontutil.numpyHstack(Stamp, time_stamp)
            	Features = Vcontutil.numpyVstack(Features, Feature)
            	Feature = []
	if len(Stamp) > 1:
        	return Features, Stamp
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
	#if os.path.exists(gmm_path + '/' + name + '.npy'):
                #gmm = Vcont.gmm_model(np.load(gmm_path + '/gmm.npz'))
                #Feature = Vcontutil.Load_Unit_Features(output_dir + '/' + ele, 0)
                print name + ' go go !!'
                Feature = Vcontutil.Load_Unit_Features(ele, 0)
                Vcontutil.fisher_vector(Feature, gmm, gmm_path + '/' + name)
        else:
                print name + '.npy already exist!'

def fisherGN_rank(inp, step = 5, overlap = 2, fps = 25):
        Q = inp[0]
        ele = inp[1]
        temp = ele.split('/')
        name = temp[len(temp) - 1]
        print name + ' go go !!'
        if os.path.isfile(fv_path + name + '.npz'):
            print 'done'
            #return ['done']
        [Feature, time_stamp] = Vcontutil.Load_Raw_Features(ele, 0)
        f = open(matching_path + name + '/matching_frame' + name + '.txt','r')
        n = 0
        s = 0
        e = 0
        for line in f:
                n += 1
                if n == 3:
                        s = int(line[11 : len(line)])
                elif n == 4:
                        e = int(line[9 : len(line)])
        if n < 3:
                return

        frame_over = overlap * fps
        frame_length = step * fps
        hi = []
        nohi = []
        boundary = []
        end = time_stamp[-1]
        num_of_interval = int(end - (frame_length - frame_over)) / frame_over
        for i in xrange(num_of_interval):
            if (i * frame_over) + frame_length < s or (i * frame_over) > e:
                nohi.append(i)
            elif (i * frame_over) + frame_length < e and (i * frame_over) > s:
                hi.append(i)
            else:
                boundary.append(i)

        th = -0.5
        while len(hi) == 0:
            th += 0.5
            if th > 300:
                return [hi, nohi, boundary]
            area = np.zeros(len(boundary))
            boundary_tmp = list(boundary)
            for i in xrange(len(boundary)):
                if s - (boundary_tmp[i] * frame_over) > 0 and s - (boundary_tmp[i] * frame_over) < th * fps:
                    hi.append(boundary_tmp[i])
                    boundary.remove(boundary_tmp[i])
                elif s - (boundary_tmp[i] * frame_over) < 0 and s - (boundary_tmp[i] * frame_over) < - th * fps:
                    hi.append(boundary_tmp[i])
                    boundary.remove(boundary_tmp[i])

        th = -0.5
        while len(nohi) == 0:
            th += 0.5
            if th > 300:
                return [hi, nohi, boundary]
            area = np.zeros(len(boundary))
            boundary_tmp = list(boundary)
            for i in xrange(len(boundary)):
                if s - (boundary_tmp[i] * frame_over) > 0 and s - (boundary_tmp[i] * frame_over) > (step - th) * fps:
                    nohi.append(boundary_tmp[i])
                    boundary.remove(boundary_tmp[i])
                elif s - (boundary_tmp[i] * frame_over) < 0 and s - (boundary_tmp[i] * frame_over) > -(step - th) * fps:
                    nohi.append(boundary_tmp[i])
                    boundary.remove(boundary_tmp[i])
        '''
        empty = []
        hi_tmp = list(hi)
        for ele in hi_tmp:
            greater = np.greater_equal(time_stamp, ele * frame_over)
            less = np.less_equal(time_stamp, ele * frame_over + frame_length)
            interval = np.equal(greater, less)
            if len(np.where(interval == True)[0]) == 0:
                hi.remove(ele)
                empty.append(ele)
        nohi_tmp = list(nohi)
        for ele in nohi_tmp:
            greater = np.greater_equal(time_stamp, ele * frame_over)
            less = np.less_equal(time_stamp, ele * frame_over + frame_length)
            interval = np.equal(greater, less)
            if len(np.where(interval == True)[0]) == 0:
                nohi.remove(ele)
                empty.append(ele)
        return [hi, nohi, boundary, empty]
        '''
        '''
        fv = []
        label = []
        q = []
        empty = []
        hi_tmp = list(hi)
        for ele in hi_tmp:
            greater = np.greater_equal(time_stamp, ele * frame_over)
            less = np.less_equal(time_stamp, ele * frame_over + frame_length)
            interval = np.equal(greater, less)
            interval1 = np.where(interval == False)[0]
            interval2 = np.where(interval == True)[0]
            if len(interval2) == 0:
                hi.remove(ele)
                empty.append(ele)
            else:
                D = np.delete(Feature, interval1, axis = 0)
                fv.append(Vcontutil.fisher_vector_rank(D, gmm))
                label.append(1)
                q.append(Q)

        nohi_tmp = list(nohi)
        for ele in nohi_tmp:
            greater = np.greater_equal(time_stamp, ele * frame_over)
            less = np.less_equal(time_stamp, ele * frame_over + frame_length)
            interval = np.equal(greater, less)
            interval1 = np.where(interval == False)[0]
            interval2 = np.where(interval == True)[0]
            if len(interval2) == 0:
                nohi.remove(ele)
                empty.append(ele)
            else:
                D = np.delete(Feature, interval1, axis = 0)
                fv.append(Vcontutil.fisher_vector_rank(D, gmm))
                label.append(0)
                q.append(Q)
        '''
        fv = []
        label = []
        q = []
        empty = []
        for ele in xrange(num_of_interval):
            greater = np.greater_equal(time_stamp, ele * frame_over)
            less = np.less_equal(time_stamp, ele * frame_over + frame_length)
            interval = np.equal(greater, less)
            interval1 = np.where(interval == False)[0]
            interval2 = np.where(interval == True)[0]
            if len(interval2) == 0:
                if ele in hi:
                    hi.remove(ele)
                elif ele in nohi:
                    nohi.remove(ele)
                elif ele in boundary:
                    boundary.remove(ele)
                empty.append(ele)
            else:
                D = np.delete(Feature, interval1, axis = 0)
                fv.append(Vcontutil.fisher_vector_rank(D, gmm))
                if ele in hi:
                    label.append(1)
                elif ele in nohi:
                    label.append(0)
                elif ele in boundary:
                    label.append(0)
                q.append(Q)

        print [hi, nohi, boundary, empty]
        print label
        np.savez(fv_path + name, fv = fv, label = label, q = q)
        return [hi, nohi, boundary, empty]

def fisherGN_rank_UW(inp):
        class_name = inp[0]
        name = inp[1]
        Q = inp[2]
        feature_path = '/media/Hao/Seagate Backup Plus Drive/HL_features/' + name
        [Feature, time_stamp] = Vcontutil.Load_Raw_Features(feature_path, 0)
        clips_path = '/media/Hao/Seagate Backup Plus Drive/UW/HL/' + class_name + '/' + name + '/test_v1/hard_label.json'
        clips = open(clips_path).read()
        clips = json.loads(clips)
        labels = clips[1]
        clip = clips[0]
        fv = []
        label = []
        q = []
        empty = []
        hi = []
        nohi = []
        boundary = []
        for i in xrange(len(labels)):
            greater = np.greater_equal(time_stamp, clip[i][0])
            less = np.less_equal(time_stamp, clip[i][1])
            interval = np.equal(greater, less)
            interval1 = np.where(interval == False)[0]
            interval2 = np.where(interval == True)[0]
            D = np.delete(Feature, interval1, axis = 0)
            fv.append(Vcontutil.fisher_vector_rank(D, gmm))
            q.append(Q)
            if len(interval2) == 0:
                empty.append(clip[i])
            else:
                if labels[i] == 1:
                    hi.append(clip[i])
                    label.append(1)
                elif labels[i] == -1:
                    nohi.append(clip[i])
                    label.append(0)
                elif labels[i] == 0:
                    boundary.append(clip[i])
                    label.append(0)

        print [hi, nohi, boundary, empty]
        np.savez(fv_path + name, fv = fv, label = label, q = q)
        return [hi, nohi, boundary, empty]

def fisherGN_Raw(ele):
        temp = ele.split('/')
        name = temp[len(temp) - 1]
        if not os.path.exists(fv_path + '/' + name + '.npy'):
                #gmm = Vcont.gmm_model(np.load(gmm_path + '/gmm.npz'))
                #Feature = Vcontutil.Load_Unit_Features(output_dir + '/' + ele, 0)
                print name + ' go go !!'
                [Feature, Stamp] = Vcontutil.Load_Raw_Features(ele, 0)
                Vcontutil.fisher_vector(Feature, gmm, fv_path + '/' + name)
        else:
                print name + '.npy already exist!'

def linear_SVM(fv, Label, C, w):
        svm = Vcontutil.linearSVM_T(fv, Label, C, w)
        return svm

def linear_pred(fv, Label, svm):
        w = svm.w()
	b = svm.bias()
	y = np.dot(fv, w) + b
	mp = sklearn.metrics.average_precision_score(Label, y)
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
	print 'mean precision: ' + str(mp)
	#acc = Vcontutil.linearSVM_P(fv, Label, svm)
	return acc, acc_pos, acc_neg, ap, mp

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
        np.save('/home/Hao/Work/fv_3_21/hmdb_training_fv', fv)
	print fv.shape
	return fv

def get_total_model(List):
	model = []
	for n in List:
		if n[-6:] == '.model':
			model_temp = mlpy.LibLinear.load_model('/home/Hao/Work/cla/' + n)
			model.append(model_temp._w())
			print model[-1]
	model = np.array(model)
	np.save('/home/Hao/Work/cla/all_model', model)

#hmdb_set = np.load('/home/Hao/Work/Cmts/hmdb_training_set.npy')
#hmdb_set = Vcontutil.numpyHstack(hmdb_set, np.load('/home/Hao/Work/Cmts/hmdb_training_set.npy'))
#f = open('/home/Hao/Work/hmdb_list.txt', 'r')
#video_list = get_video_list(f, video_list)
#index = np.delete(range(len(video_list)), hmdb_set)
#video_list = np.delete(video_list, index)
#input_list = []
#for i in video_list:
#        input_list.append('/media/Hao/My Book/hmdb_total_fv/' + i[18:])

#get_total_fv(input_list)
#model_dir = os.listdir('/home/Hao/Work/cla/')
#get_total_model(model_dir)
#sys.exit()

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
		#name = cluster[0]
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
			print svm[i].w()
			print svm[i].bias()
			sys.exit()
		if count > 1:
			acc_avg = float(sum(accuracy[: , 4])) / accuracy.shape[0]
			index = np.where(accuracy == max(accuracy[: , 0]))[0][0]
		else:
			acc_avg = accuracy[4]
			index = 0
		print 'cross-validation mean average precision: ' + str(acc_avg)
		#svm[index].save_model(cla_path + name + '/' + name + '_' + str(len(cluster)) + '.model')
		np.save(cla_path + name + '/' + name + '_' + str(len(cluster)) + '_accuracy', accuracy)
	else:
		return
	print 'Time cost: ' + str(round(time.time() - sTime, 3)) + 'second'

def get_tfidf(tfidf, group, K, method = 0): #method = 0: top 10 of top 10, method = 1: directly sum
	bow = np.zeros(tfidf.shape[1])
	for ele in group:
		ele = int(ele)
		if method == 0:
			tmp = heapq.nlargest(K, enumerate(tfidf[ele]), key=lambda x:x[1])
			for i in range(K):
				bow[tmp[i][0]] += tfidf[ele][tmp[i][0]]
		else:
			bow += tfidf[ele]

	top_terms = heapq.nlargest(K, enumerate(bow), key=lambda x:x[1])
	return top_terms

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
	xx = 0
	for i in groups:
		num = len(i)
		if num < 1:
			break
		#name = get_seed_name(i[0])
		name = str(i[0])
		#temp = np.load(cla_path + name + '/' + name + '_' + str(num) + '_accuracy.npy')
		group_top_terms = get_tfidf(tfidf_bow, i, term_num)
		group_top_terms = get_tuple_index(group_top_terms)
		xx += 1
		print 'case : ' + str(xx)
		print group_top_terms
		term_acc = 0
		#for j in i:
		'''
		for j in query_video:
			query_top_terms = get_tfidf(tfidf_bow, [j], term_num)
			query_top_terms = get_tuple_index(query_top_terms)
			for k in group_top_terms:
				if k in query_top_terms:
					term_acc += 1. / term_num
		term_acc /= num
        	print term_acc
		'''
		'''
		information = np.load(cla_path + name + '/' + name + '_' + str(num) + '_accuracy.npy')
        	if num > 2:
			mp = np.mean(information[:,-1])
		else:
			mp = information[-1]
		print name, num, term_acc, mp
		'''
		#en = compute_entropy(bow_g)
		data = Vcontutil.numpyHstack(name, num)
		#data = Vcontutil.numpyHstack(data, float(sum(temp[:, 0])) / temp.shape[0])
		data = Vcontutil.numpyHstack(data, term_acc)
		#data = Vcontutil.numpyHstack(data, mp)
		if Fir == True:
			D = data
			#B = bow_g
			group_top_term = group_top_terms
			Fir = False
		else:
			D = Vcontutil.numpyVstack(D, data)
			#B = Vcontutil.numpyVstack(B, bow_g)
			group_top_term = Vcontutil.numpyVstack(group_top_term, group_top_terms)
	return D, group_top_term

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
                bow_g = get_tfidf(tfidf_bow, i[1 : len(i)], term_num)
		entries = np.delete(range(mid_numbers) , get_number_info_from_group_data(i[1 : len(i)], 0, True))
                for j in entries:
                        bow_v = get_tfidf(tfidf_bow, [j], term_num)
                        acc = Compute_term_acc(np.where(bow_v >= np.sort(bow_v)[np.where(np.sort(bow_v) != 0)[0][0]])[0], np.where(bow_g >= np.sort(bow_g)[np.where(np.sort(bow_g) != 0)[0][0]])[0])
			if acc > term_acc[k]*1.5:
				groups[k].append(str(j))
		#print i[1 : len(i)]
		#print groups[k]
		print len(i), len(groups[k])
		size.append([len(i), len(groups[k])])
	return groups, size

def get_overlap(A, B):
	if len(B) > len(A):
		temp = A
		A = B
		B = temp
	xx = 0
	for i in B:
		if i in A:
			xx += 1

	return float(xx) / (len(A) + len(B) - xx)

def train_term(cluster):
	i = cluster[1]
	name = get_seed_name(cluster[0])
	clu = np.array(cluster[2:], dtype = np.int)
	print '\n####### ' + name + ' #######'
	#if os.path.isfile(cla_path + name + '_' + str(i) + '_' + '.model'):
        #	print 'done'
	#	return
	fv_pos = convert_index2fv(clu, fv_mid_total)
	label_pos = np.ones(fv_pos.shape[0])
        fv_neg = fv_hmdb_training
        label_neg = np.zeros(fv_neg.shape[0])
	fv = Vcontutil.numpyVstack(fv_pos, fv_neg)
        label = Vcontutil.numpyHstack(label_pos, label_neg)
	w = {0 : 1, 1 : 1}
	svm = linear_SVM(fv, label, 100, w)

	#fv_pos = fv_mid_testing
        #label_pos = np.ones(fv_pos.shape[0])
        #fv_neg = fv_hmdb_testing
        #label_neg = np.zeros(fv_neg.shape[0])
        #fv = Vcontutil.numpyVstack(fv_pos, fv_neg)
        #label = Vcontutil.numpyHstack(label_pos, label_neg)
        #acc = linear_pred(fv, label, svm)
        #accuracy = np.array(acc)

        svm.save_model(cla_path + name + '_' + str(i) + '_' + '.model')
	#svm.save_model('/home/Hao/Work/cla/')
	#if not os.path.isdir(cla_path + 'acc/'):
	#	subprocess.call('mkdir ' + cla_path + 'acc/', shell = True)
	#np.save(cla_path + 'acc/' + name + '_' + str(i) + '_accuracy', accuracy)

def get_score(fvs, model):
	a = np.ones(len(fvs))
	a.shape = len(fvs), 1
	fvs = Vcontutil.numpyHstack(a, fvs)
	scores = np.dot(fvs, model.T)
	return scores
'''
f = open('/home/Hao/Work/match_list.txt', 'r')
video_list = get_video_list(f, video_list)
raw_test = np.load('/home/Hao/Work/raw_testing_set.npy')
raw_test2 = np.load('/home/Hao/Work/Cmts/raw/raw_testing_set2_4_7.npy')
video = []
fw = open('/home/Hao/Work/raw_check_list2_4_9.txt', 'w')
for i in xrange(len(video_list)):
    if i in raw_test2 and i not in raw_test:
        video.append([video_list[i][33:], i])
        fw.write(video_list[i][33:] + '  ' + str(i) + '\n')
        print video[-1]
'''

'''
### Fisher Vector Verification
#mid_set = np.load('/home/Hao/Work/Cmts/hmdb_training_set.npy')
f = open('/home/Hao/Work/mid_list.txt', 'r')
video_list = get_video_list(f, video_list)
dirs = '/media/Hao/My Book/mid_total_fv/'
dirs = '/home/Hao/Work/fv/'
fv_mid_total = np.load('/home/Hao/Work/mid_total_fv.npy')
for i in xrange(len(video_list)):
	print 'testing case ' + str(i) + ': ' + str(video_list[i][18:])
	path = dirs + video_list[i][18:]
	fv = np.load(path)
	error = fv - fv_mid_total[i]
print sum(error)
### Score Evaluation
#fv_mid_total = np.load('/home/Hao/Work/fv_3_21/mid_total_fv.npy')
#all_model = np.load('/home/Hao/Work/cla/all_model.npy')
#scores = get_score(fv_mid_total, all_model)
#np.save('/home/Hao/Work/mid_total_scores', scores)
#sys.exit()
'''
'''
## Verification
mid_total_scores = np.load('/home/Hao/Work/mid_total_scores.npy')
mid_testing_set = np.load('/home/Hao/Work/mid_testing_set.npy')
group_total = np.load('/home/Hao/Work/500_group_selection.npy')
tfidf_bow = np.load('/home/Hao/Work/Cmts/tfidf_bow.npy')
idf = np.load('/home/Hao/Work/Cmts/mid_idf.npy')
acc = 0
for i in xrange(len(mid_testing_set)):
	print 'testing case ' + str(i) + ': ' + str(mid_testing_set[i])
	tfidf_target = get_tfidf(tfidf_bow, [mid_testing_set[i]], 10)
	scores = mid_total_scores[mid_testing_set[i]]
	for j in xrange(len(group_total)):
		if str(mid_testing_set[i]) in group_total[j]:
			group = group_total[j]
			tfidf_pred = get_tfidf(tfidf_bow, group[1:], 10)
			break
	ten_target = heapq.nlargest(10, enumerate(tfidf_target), key=lambda x:x[1])
	ten_pred = heapq.nlargest(10, enumerate(tfidf_pred), key=lambda x:x[1])
	print ten_target
	print ten_pred
	count = 0
	for j in xrange(len(ten_target)):
		if ten_target[j][0] in ten_pred[:][0]:
			count += 1
	acc += count / 10.
	print 'acc :' + str(count / 10.)
acc /= len(mid_testing_set)
print acc
'''
#f = open('/home/Hao/Work/mid_list.txt', 'r')
#video_list = get_video_list(f, video_list)
#index = np.delete(range(len(video_list)), mid_set)
#video_list = np.delete(video_list, index)
#input_list = []
#for i in video_list:
#        input_list.append('/home/Hao/Work/hmdb_features_fix360/' + i[18 : -4])


'''
### mid video set extracting
## Load Cluster by seeds seletion
clu_path = '/home/Hao/Work/Cmts/cmt_clu4.txt'
clu_file = open(clu_path, 'r')
cluster_temp = clu_file.read()
cluster = cluster_temp.split('\n')

## Initial path setting
cla_path = '/home/Hao/Work/cla/'

## Get Cluster Group
clu_group = get_clu(cluster, cla_path)

## Get tfidf count
tfidf_bow = np.load('/home/Hao/Work/Cmts/tfidf_bow.npy')
tfidf_count= np.load('/home/Hao/Work/Cmts/tfidf_count.npy')
idf = np.load('/home/Hao/Work/Cmts/mid_idf.npy')

## Get group data [name, length, mP, entropy]
#group_data = np.load('/home/Hao/Work/group_data.npy')
#clu_group = np.load('/home/Hao/Work/group_expand.npy')
#remove = []
#for i in range(len(clu_group)):
#	if len(clu_group[i]) < 6:
#		remove.append(i)
#clu_group = np.delete(clu_group, remove)
test_set = np.load('/home/Hao/Work/mid_testing_set.npy')
test_set = np.sort(test_set)

[D, group_top_term] = Load_mAP(clu_group)
np.save('/home/Hao/Work/debug/group_CHISQR_10_top_term', group_top_term)

group_top_term = np.load('/home/Hao/Work/debug/group_CHISQR_10_top_term.npy')
acc_max = []
len_group = []
for i in xrange(len(test_set)):
	test_top_term = get_tfidf(tfidf_bow, [test_set[i]], 10)
	test_top_term = get_tuple_index(test_top_term)
	acc_group = []
	for j in xrange(len(clu_group)):
		if str(i) in clu_group[j]:
			acc = 0
			for k in test_top_term:
				if k in group_top_term[j]:
					acc += 1. / 10
			acc_group.append(acc)
	acc_max.append(max(acc_group))
	len_group.append(len(clu_group[acc_group.index(acc_max[-1])]))
	print 'case ' + str(i + 1) + ': acc ' + str(acc_max[-1]) + ', size ' + str(len_group[-1])
np.save('/home/Hao/Work/debug/test_group_CHISQR_10_term_acc', [acc_max, len_group])
'''
'''
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
#clu_group = get_clu(cluster[0 : len(cluster) - 2], cla_path)
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
#clu_group = extract_list(clu_group, term_acc)
clu_group = np.load('/home/Hao/Work/500_group_selection.npy')
testing_set = np.load('/home/Hao/Work/mid_testing_set.npy')
#testing_set = np.array(testing_set, np.str)

training_group = []
xx = 0
for i in clu_group:
	#name = '_' + i[0][:-16]
	temp = [i[0], xx]
	xx += 1
	#temp = []
	for j in i[1:]:
		if not int(j) in testing_set:
			temp.append(j)
		#else:
			#print testing_set[np.where(testing_set == int(j))[0]], int(j)
	#sys.exit()
	#print len(i)
	#print len(temp)
	training_group.append(temp)
	#if not os.path.exists(cla_path + name):
        #	subprocess.call('mkdir ' + cla_path + name, shell = True)

#xx = np.zeros(842)
#for i in training_group:
#	for j in i:
#		print j
#		xx[j] = 1
#print sum(xx)
#sys.exit()
#remove = []
#for i in range(len(clu_group)):
#        if len(clu_group[i]) < 6:
#                remove.append(i)
#clu_group = np.delete(clu_group, remove)

#coverage = np.zeros(len(cmtz))
#for i in clu_group:
#	for j in range(len(i) - 1):
#		j += 1
#		coverage[int(i[j])] = 1
#c = 0
#for i in coverage:
#	if i == 1:
#		c += 1
##print float(c) / len(cmtz)

## Load prepared fv
#fv_mid = np.load('/home/Hao/Work/mid_total_fv.npy')
#fv_hmdb = np.load('/home/Hao/Work/hmdb_0.4_fv.npy')

#fv_mid_training = np.load('/home/Hao/Work/fv_3_21/mid_training_fv.npy')
#fv_mid_testing = np.load('/home/Hao/Work/fv_3_21/mid_testing_fv.npy')
fv_mid_total = np.load('/home/Hao/Work/fv_3_21/mid_total_fv.npy')
#fv_hmdb_training = np.load('/home/Hao/Work/hmdb_0.4_fv.npy')
fv_hmdb_training = np.load('/home/Hao/Work/fv_3_21/hmdb_training_fv.npy')
fv_hmdb_testing = np.load('/home/Hao/Work/fv_3_21/hmdb_testing_fv.npy')

#for i in training_group:
#	if len(i) == 1:
#		print i
#print training_group
#train_term(training_group[0])
#for i in range(len(training_group)):
#	train_term(teaining_group[i],i)
#print len(training_group)
#train_term(training_group[0])
p = Pool(4)
p.map(train_term, training_group)
#cross_validation(clu_group[1])
#sys.exit()
#p = Pool(2)
#p.map(cross_validation, training_group[:335])

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
'''
### term accuracy & mean precision & coverage compute, testing group selection
clu_group = np.load('/home/Hao/Work/group_expand.npy')
cla_path = '/home/Hao/Work/cla_3_15/'

fv_mid = np.load('/home/Hao/Work/mid_total_fv.npy')
fv_hmdb = np.load('/home/Hao/Work/hmdb_0.4_fv.npy')
label_neg = np.zeros(len(fv_hmdb))

tfidf_bow = np.load('/home/Hao/Work/bowtfidf_bow.npy')
tfidf_count = np.load('/home/Hao/Work/Cmts/tfidf_count.npy')
idf = np.load('/home/Hao/Work/Cmts/mid_idf.npy')

#D = Load_mAP(clu_group)
D = np.load('/home/Hao/Work/group_information.npy')

length = np.array(D[:, -3], dtype = np.float32)
term_acc = np.array(D[:, -2], dtype = np.float32)
mp = np.array(D[:, -1], dtype = np.float32)

short_index = np.where(length == 2)[0]
length = np.delete(length, short_index)
term_acc = np.delete(term_acc, short_index)
mp = np.delete(mp, short_index)
clu_group = np.delete(clu_group, short_index)
#overlap = np.zeros([len(clu_group), len(clu_group)])
#for i in range(len(clu_group)):
#	for j in range(len(clu_group)):
#		overlap[i,j] = get_overlap(clu_group[i], clu_group[j])
#		print overlap[i, j]
#np.save('/home/Hao/Work/group_overlap', overlap)
overlap = np.load('/home/Hao/Work/group_overlap.npy')

th = 0.805
mp_sort = np.sort(mp)
max_1 = 0
index = [0]
for i in range(len(mp_sort)):
	if mp[i] > 0.9999999:
		max_1 = max(max_1, length[i])
		if max_1 == length[i]:
			index[0] = i
			overlapping = overlap[i]
			original_index = range(len(mp))
			delete_index = np.array([i])

while len(delete_index) < (len(mp) - 1):
	delete_index = Vcontutil.numpyHstack(delete_index, np.where(overlapping > th)[0])
	original_index = np.delete(range(len(mp)) , delete_index)
	overlapping = np.delete(overlapping , delete_index)
	mp_eva = np.delete(mp, delete_index)
	mp_max = max(mp_eva)
	#print mp_max
	index.append(original_index[np.where(mp_eva == mp_max)[0][0]])
	delete_index = Vcontutil.numpyHstack(delete_index, index[-1])
	overlapping = overlap[index[-1]]
	print index[-1]
print len(index)
index = np.array(index, dtype = np.int)
delete_index = np.delete(range(len(mp)), index)
clu_save = np.delete(clu_group, delete_index)
#np.save('/home/Hao/Work/500_group_selection', clu_save)

lucky = []
while len(lucky) < 200:
	mylist = []
	for j in index:
		for k in range(len(clu_group[j])):
			if k == 0 or clu_group[j][k] in lucky:
				continue
			mylist += [clu_group[j][k]] * len(clu_group[j])
	lucky.append(random.choice(mylist))
	for j in index:
		temp = list(clu_group[j])
		for k in lucky:
			if k in temp:
				temp.remove(k)
		if len(temp) == 1:
			lucky.remove(lucky[-1])
			break
	print lucky[-1]
testing_group = np.array(lucky, dtype = np.int)
training_group = np.delete(range(mid_numbers), testing_group)
print len(training_group), len(testing_group)
#np.save('/home/Hao/Work/mid_testing_set', testing_group)
#np.save('/home/Hao/Work/mid_training_set', training_group)

#clu_group = np.load('/home/Hao/Work/500_group_selection.npy')
con = np.zeros(mid_numbers)
for i in clu_save:
	xx = i[1:]
	for j in xx:
		con[int(j)] = 1
converage = np.mean(con)
print 'total coverage: ' + str(converage)

testing_group = np.load('/home/Hao/Work/mid_testing_set.npy')
training_group2 = []
for i in xrange(len(con)):
	if con[i] == 1 and i not in testing_group:
		training_group2.append(i)
training_group2 = np.array(training_group2)
print 'training set coverage: ' + str(float(len(training_group2)) / 642)
#np.save('/home/Hao/Work/mid_training_set2', training_group2)

con = np.zeros(len(clu_group))
for i in range(len(clu_group)):
        for j in clu_group[i][1:]:
                if int(j) in testing_group:
                        con[i] = 1
                        break
print 'group coverage: ' + str(np.mean(con))
'''
'''
## merge different training computer results

cla_path = '/home/Hao/Work/cla_3_15/'
cla_path0 = '/home/Hao/Work/cla_server_3_15/'
cla_path1 = '/home/Hao/Work/cla_Hertz_3_15/'
cla_path2 = '/home/Hao/Work/cla_lab_3_15/'
group_acc = []
group_pre = []
xx = 0
yy = 0
zz = 0
for i in clu_group:
	name = get_seed_name(i[0])
	if os.path.isfile(cla_path + name + '/' + name + '_' + str(len(i)) + '_accuracy.npy') == False:
		xx += 1

		if os.path.isfile(cla_path0 + name + '/' + name + '_' + str(len(i)) + '_accuracy.npy') == True:
			xx += 1
			group_acc.append(i)
			subprocess.call('cp -r ' + cla_path0 + name + '/ ' + cla_path + name + '/',shell = True)
			continue
		if os.path.isfile(cla_path1 + name + '/' + name + '_' + str(len(i)) + '_accuracy.npy') == True:
                	yy += 1
                	group_acc.append(i)
                	subprocess.call('cp -r ' + cla_path1 + name + '/ ' + cla_path + name + '/',shell = True)
                	continue
		if os.path.isfile(cla_path2 + name + '/' + name + '_' + str(len(i)) + '_accuracy.npy') == True:
                	zz += 1
                	group_acc.append(i)
                	subprocess.call('cp -r ' + cla_path2 + name + '/ ' + cla_path + name + '/',shell = True)
                	continue

	#temp = np.load(cla_path + name + '/' + name + '_' + str(len(i)) + '_accuracy.npy')
	#print temp
	#group_acc.append(float(sum(temp[:, 0])) / len(temp))
	#group_pre.append(float(sum(temp[:, 3])) / len(temp))
#print group_acc[1]
group_acc = np.array(group_acc)
print xx, yy, zz
print len(group_acc)
#np.save('/home/Hao/Work/group_r', group_acc)
sys.exit()
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
hmdb_set = np.load('/home/Hao/Work/Cmts/hmdb_training_set.npy')
f = open('/home/Hao/Work/hmdb_list.txt', 'r')
video_list = get_video_list(f, video_list)
index = np.delete(range(len(video_list)), hmdb_set[0:2700])
video_list = np.delete(video_list, index)
input_list = []
for i in video_list:
	input_list.append('/home/Hao/Work/hmdb_features_fix360/' + i[18 : -4])
#size = check_features_size(input_list)
#np.save('/home/Hao/Work/hmdb_training_ft_size', size)
Features = get_features(input_list, 679)
#print Features.shape

mid_set = np.load('/home/Hao/Work/mid_training_set.npy')
f = open('/home/Hao/Work/mid_list.txt', 'r')
video_list = []
video_list = get_video_list(f, video_list)
index = np.delete(range(len(video_list)), mid_set)
video_list = np.delete(video_list, index)
input_list = []
for i in video_list:
        input_list.append('/media/Hao/My Book/mid_features_fix360/' + i[18 : -4])
#size = check_features_size(input_list)
#np.save('/home/Hao/Work/mid_training_ft_size', size)
Features = Vcontutil.numpyVstack(Features, get_features(input_list, 22842))

raw_set = np.load('/home/Hao/Work/raw_training_set.npy')
f = open('/home/Hao/Work/match_list.txt', 'r')
video_list = []
video_list = get_video_list(f, video_list)
index = np.delete(range(len(video_list)), raw_set)
video_list = np.delete(video_list, index)
input_list = []
for i in video_list:
        input_list.append('/media/Hao/My Book/raw_features/' + i[32:])
#[Features, time_stamp] = Vcontutil.Load_Raw_Features(input_list[-1], 0)
#print Features[0]
#print time_stamp
#print Features.shape
#print time_stamp.shape
size = check_features_size(input_list, 0, 1)
np.save('/home/Hao/Work/raw_training_ft_size', size)
[fv_temp, garbage] = get_features(input_list, 12180, 1)
Features = Vcontutil.numpyVstack(Features, fv_temp)
#print Features.shape
gmm_path = '/home/Hao/Work/gmm/'
mid_gmm(Features, gmm_path, 256, 4)
'''
'''
### Fisher Vector encoding
gmm = Vcont.gmm_model(np.load('/home/Hao/Work/gmm/gmm.npz'))
#hmdb_set = np.load('/home/Hao/Work/Cmts/hmdb_testing_set.npy')
#hmdb_set = Vcontutil.numpyHstack(hmdb_set, np.load('/home/Hao/Work/Cmts/hmdb_training_set.npy'))
#f = open('/home/Hao/Work/debug/match_list.txt', 'r')
f = open('/home/Hao/Work/Cmts/match_list_backup.txt', 'r')
video_list = get_video_list(f, video_list)
#index = np.delete(range(len(video_list)), hmdb_set)
#video_list = np.delete(video_list, index)
matching_path = '/media/Hao/My Book/raw/'
#raw_test_set = np.load('/home/Hao/Work/Cmts/raw/raw_testing_set3_4_10.npy')
input_list = []
for i in range(len(video_list)):
        #input_list.append([i, '/media/Hao/My Book' + video_list[i][18:]])
        input_list.append([i, '/media/Hao/My Book/raw_features/' + video_list[i]])
        #input_list.append('/media/Hao/My Book/raw_features/' + video_list[i])
        print input_list[-1]
#gmm_path = '/media/Hao/My Book/debug'
fv_path = '/media/Hao/My Book/raw_whole_fv_demo/'
'''
'''
rank_interval = []
for ele in input_list:
    #if ele[0] == 282:
        print ele[1]
        rank_interval.append(fisherGN_rank(ele))
        print rank_interval[-1]
'''
'''
p = Pool(4)
#fisherGN_rank(input_list[0])
rank_interval = p.map(fisherGN_rank, input_list)
#p.map(fisherGN_Raw, input_list)
#np.save('/media/Hao/My Book/debug/rank_interval', rank_interval)
'''
'''
mid_set = np.load('/home/Hao/Work/Cmts/hmdb_training_set.npy')
f = open('/home/Hao/Work/hmdb_list.txt', 'r')
video_list = []
video_list = get_video_list(f, video_list)
index = np.delete(range(len(video_list)), mid_set)
video_list = np.delete(video_list, index)
input_list = []
for i in video_list:
        input_list.append('/home/Hao/Work/hmdb_features_fix360/' + i[18 : -4])

gmm_path = '/media/Hao/My Book/hmdb_training_fv'
p = Pool(4)
p.map(fisherGN, input_list)
'''
'''
#f = open('/home/al-farabi/Desktop/hmdb_list.txt', 'r')
gmm_path = '/media/Hao/My Book/hmdb_testing_fv'
#video_list = get_video_list(f, video_list)
p = Pool(4)
p.map(fisherGN, input_list)

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
