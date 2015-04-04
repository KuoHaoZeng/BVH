import numpy as np
from numpy import linalg
from scipy import ndimage, signal, stats
from multiprocessing import Pool
import sys, math, copy, time, cv2, os, heapq, Vcontutil, Vcont, mlpy, random, sklearn.metrics, subprocess

def get_tuple_index(arr):
        index = np.zeros(len(arr))
        for i in xrange(len(arr)):
                index[i] = arr[i][0]
        return index

def get_clu(cluster):
        name = []
        clu_group = []
        for ele in cluster:
                temp = ele.split(' ')[0]
                if temp[len(temp) - 3 : len(temp)] == 'bow':
                        a = temp.find('all')
                        name.append('_' + temp[0 : a])
                clu_group.append(ele.split(' ')[0 : len(ele.split(' ')) - 1])
        return clu_group

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

def acc(A, B):
    acc = 0
    for ele in A:
        if ele in B:
            acc += 1. / len(A)
    return round(acc, 3)

def linear_pred(fv, Label, svm):
    w = svm.w()
    b = svm.bias()
    y = np.dot(fv, w) + b
    mp = sklearn.metrics.average_precision_score(Label, y)
    '''
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
    #print 'average accuracy: ' + str(acc)
    #print 'positive accuracy: ' + str(acc_pos)
    #print 'negative accuracy: ' + str(acc_neg)
    #print 'average precision: ' + str(ap)
    '''
    print 'mean precision: ' + str(mp)
    #acc = Vcontutil.linearSVM_P(fv, Label, svm)
    return mp

def compute_term_acc(group):
	id_group = group[-1]
	print 'case ID: ' + str(id_group)
	group = group[0:-1]
	group_top_terms = get_tfidf(tfidf_bow, group, 10)
        group_top_terms = get_tuple_index(group_top_terms)
	acc_group = []
	for i in xrange(len(group)):
		query_top_terms = get_tfidf(tfidf_bow, group[i], 10)
        query_top_terms = get_tuple_index(query_top_terms)
        acc = 0
        for j in query_top_terms:
            if j in group_top_terms:
                acc += 1. / 10
		acc_group.append(round(acc, 3))
	return [acc_group, id_group]

def get_greedy_group(data):
    ## data = [seed ID, seed]
    ## return [seed ID, generated groups, maximum term accuracy, mean term accuracy,
    ##         minimum term accuracy, 25% term accuracy]
    ID = data[0]
    seed = data[1]
    print 'seed ID: ' + str(ID)
    ## compute top terms index for each video first
    top_terms = []
    print 'computing top terms for each video ...'
    for i in xrange(842):
        tmp_top_terms = get_tfidf(tfidf_bow, [i], 10)
        tmp_top_terms = get_tuple_index(tmp_top_terms)

        # top_terms = (842, 10)
        top_terms.append(tmp_top_terms)

    group = [seed]
    acc_mean = []
    acc_max = []
    acc_min = []
    acc_25 = []
    ## start
    for i in xrange(841):
        print 'process: ' + str(i)
        group_top_terms = get_tfidf(tfidf_bow, group, 10)
        group_top_terms = get_tuple_index(group_top_terms)
        for j in group_top_terms:
            print tfidf_term[j]

        ## compute group mean terms accuracy over each member
        acc_group = []
        for j in group:
            acc = 0
            for k in top_terms[j]:
                if k in group_top_terms:
                     acc += 1. / 10
            acc_group.append(round(acc, 3))
        if np.mean(acc_group) < 0.3:  ## Break as mean terms accuracy lower than 0.3
            group.remove(group[-1])
            break
        acc_mean.append(np.mean(acc_group))
        acc_max.append(max(acc_group))
        acc_min.append(min(acc_group))
        acc_25.append(stats.scoreatpercentile(acc_group, 25))
        print acc_mean[-1], acc_max[-1], acc_min[-1], acc_25[-1]

        ## selection new group member by highest terms overlapping between group and each candidate
        acc_tmp = []
        for j in xrange(842):
            if j in group: # if j already in group, append min to acc_tmp so that it won't be choosed
                acc_tmp.append(-sys.maxint)
            else:
                acc = 0
                for term in top_terms[j]:
                    if term in group_top_terms:
                        acc += 1. / 10
                acc_tmp.append(round(acc, 3))
        max_index = np.argmax(acc_tmp)
        group.append(max_index)
    return [ID, group, acc_max, acc_mean, acc_min, acc_25]

def get_whole_group(seed):
    group = [seed]
    group_top_terms = get_tfidf(tfidf_bow, group, 10)
    group_top_terms = get_tuple_index(group_top_terms)
    for i in xrange(841):
        print 'process: ' + str(i)
        acc_search = np.zeros(842)
        for j in xrange(842):
            if j in group:
                continue
            group.append(j)
            group_top_terms = get_tfidf(tfidf_bow, group, 10)
            group_top_terms = get_tuple_index(group_top_terms)
            acc_group = []
            for k in group:
                query_top_terms = get_tfidf(tfidf_bow, [k], 10)
                query_top_terms = get_tuple_index(query_top_terms)
                acc = 0
                for l in query_top_terms:
                    if l in group_top_terms:
                        acc += 1. / 10
                acc_group.append(round(acc, 3))
            acc_search[j] = np.mean(acc_group)
            group.remove(j)
        max_acc = max(acc_search)
        print max_acc
        group.append(np.where(acc_search == max_acc)[0][0])
        print group
        group_top_terms = get_tfidf(tfidf_bow, group, 10)
        group_top_terms = get_tuple_index(group_top_terms)
        for j in group_top_terms:
            print tfidf_term[j]

def group_analysis(data):
    ## data = [ID, [[group A], [group B], ... ]]
    ## retrun [ID, maximum overlapping ratio, maximum overlapping group ID]
    ID = data[0]
    groups = data[1]
    print 'seed ID: ' + str(ID)

    overlap = [0] * len(groups)
    for i in xrange(len(groups)):
        ## if there is not seed itself, compute overlap between group[i] and group[ID]
        if i != ID:
            tmp = 0.
            for ele in groups[i]:
                if ele in groups[ID]:
                    tmp += 1
            overlap[i] = round(tmp / len(groups[ID]), 3)
    print 'max overlap: ' + str(max(overlap))
    return ID, max(overlap), overlap.index(max(overlap))

def test_2methods(data):
    ## data = [ID, group(ID)]
    ## return [ID, 3-folder cross-validation mean average precision]
    ID = data[0]
    group = data[1]
    print 'seed ID: ' + str(ID)

    ## if folder(ID) does not exist, create it
    path = '/home/Hao/Work/mAP_verify/' + str(ID)
    if not os.path.isdir(path):
        subprocess.call('mkdir ' + path, shell = True)
    if not os.path.isdir(path + '/' + str(len(group))):
        subprocess.call('mkdir ' + path + '/' + str(len(group)), shell = True)


    ## get negative set expan by adding term accuracy = 0 video
    group_top_terms = get_tfidf(tfidf_bow, group, 10)
    group_top_terms = get_tuple_index(group_top_terms)
    neg_expan = []
    for i in xrange(842):
        if i not in group:
            query_top_terms = get_tfidf(tfidf_bow, [i], 10)
            query_top_terms = get_tuple_index(query_top_terms)
            acc = 0
            for l in query_top_terms:
                if l in group_top_terms:
                    acc += 1. / 10
            if acc == 0:
                neg_expan.append(i)

    ## k-folder initialition
    k = len(group)
    pos = np.delete(range(842), group)
    fv_pos = np.delete(fv_mid, pos, axis = 0)
    neg = np.delete(range(842), neg_expan)
    fv_neg_mid = np.delete(fv_mid, neg, axis = 0)
    fv_neg_hmdb = fv_hmdb_train
    idx_pos = mlpy.cv_kfold(len(fv_pos), k)
    idx_neg_mid = mlpy.cv_kfold(len(neg), k)
    idx_neg_hmdb = mlpy.cv_kfold(len(fv_hmdb_train), k)
    ## save k-folder information for back review
    np.save(path + '/' + str(len(group)) + '/idx_pos', idx_pos)
    np.save(path + '/' + str(len(group)) + '/idx_neg_mid', idx_neg_mid)
    np.save(path + '/' + str(len(group)) + '/idx_neg_hmdb', idx_neg_hmdb)
    mp = []
    for i in xrange(k):
        ## k-folder cross-validation
        fv_pos_cross = np.delete(fv_pos, idx_pos[i][1], axis = 0)
        fv_neg_mid_cross = np.delete(fv_neg_mid, idx_neg_mid[i][1], axis = 0)
        fv_neg_hmdb_cross = np.delete(fv_neg_hmdb, idx_neg_hmdb[i][1], axis = 0)
        fv_neg_cross = np.vstack((fv_neg_mid_cross, fv_neg_hmdb_cross))
        #fv_neg_cross = fv_neg_hmdb_cross
        label_pos = np.ones(len(fv_pos_cross))
        label_neg = np.zeros(len(fv_neg_cross))
        fv = np.vstack((fv_pos_cross, fv_neg_cross))
        label = np.hstack((label_pos, label_neg))
        svm = Vcontutil.linearSVM_T(fv, label, 10, {0 : 1, 1 : 1})
        svm.save_model(path + '/' + str(len(group)) + '/' + str(i) + '.model')

        ## compute mean average precision on testing set
        fv_pos_cross = np.delete(fv_pos, idx_pos[i][0], axis = 0)
        fv_neg_mid_cross = np.delete(fv_neg_mid, idx_neg_mid[i][0], axis = 0)
        fv_neg_hmdb_cross = np.delete(fv_neg_hmdb, idx_neg_hmdb[i][0], axis = 0)
        fv_neg_cross = np.vstack((fv_neg_mid_cross, fv_neg_hmdb_cross))
        #fv_neg_cross = fv_neg_hmdb_cross
        label_pos = np.ones(len(fv_pos_cross))
        label_neg = np.zeros(len(fv_neg_cross))
        fv = np.vstack((fv_pos_cross, fv_neg_cross))
        label = np.hstack((label_pos, label_neg))
        mp.append(linear_pred(fv, label, svm))
    return ID, round(np.mean(mp), 3)

def des(ID, mean_acc):
    ## [ID, mean_acc of group(ID)]
    ## return [ID, the slope of described function = (f(mean_acc) = group_member)', y-axis]
    x = [mean_acc[0]]
    y = [0]
    step = []
    for i in xrange(len(mean_acc) - 1):
        ## let function be a monotonously rising function
        if x[-1] > mean_acc[i + 1]:
            idx = np.where(mean_acc == x[-1])[0][0]
            x.append(mean_acc[i + 1])
            y.append(i + 1)
            step.append(i + 1 - idx)
    ## compute slope of this function
    z = []
    for i in xrange(len(x) - 1):
        z.append(((1 - x[i + 1]) - (1 - x[i])) / step[i])
    ## apply guassian filter on this slope function
    z = ndimage.filters.gaussian_filter(z, 0.2)
    return ID, z, y

def level_set(data):
    ## data = [ID, [acc_mean of group(ID)]]
    ## return [ID, Each level set selection end index of group(ID)]
    ID = data[0]
    mean_acc = data[1]
    level_end = []
    print 'seed ID: ' + str(ID)

    ## if size of group greater than 5, do level set selection.
    if len(mean_acc) > 5:
        ## get the slope of described function, (f(mean_acc) = group_member)'
        [ID, z, y] = des(ID, mean_acc)
        ## get local minimum of this function
        local_min = signal.argrelextrema(z, np.less)[0]
        ## find each levels end index in group(ID)
        for ele in local_min:
            level_end.append(y[ele - 1])
    return ID, level_end

def level_expan(data):
    ## data = [ID, [group(ID)], level_end], group(ID) need to be expaned
    ## return [ID, new group]
    ID = data[0]
    group = data[1]
    level_end = data[2]
    print 'seed ID: ' + str(ID)

    groups = []
    for ele in level_end:
        groups.append(group[0 : ele])
    groups.append(group)

    return ID, groups

def oracle(data):
    ## data = [testing ID, all groups set]
    ## return [testing ID, maximum term accuracy]
    ID = data[0]
    groups = data[1]
    print 'test ID: ' + str(ID)

    query_top_terms = get_tfidf(tfidf_bow, [ID], 10)
    query_top_terms = get_tuple_index(query_top_terms)
    term_acc = []
    for ele in groups:
        if ID in ele:
            group = list(ele)
            group.remove(ID)
            group_top_terms = get_tfidf(tfidf_bow, group, 10)
            group_top_terms = get_tuple_index(group_top_terms)
            term_acc.append(acc(query_top_terms, group_top_terms))
    return ID, max(term_acc)

###############################################################################









#clu_path = '/home/Hao/Work/Cmts/cmt_clu6.txt'
#clu_file = open(clu_path, 'r')
#cluster_temp = clu_file.read()
#cluster = cluster_temp.split('\n')
#clu_group = get_clu(cluster)

#for i in xrange(len(clu_group)):
#	clu_group[i].append(i)

tfidf_bow = np.load('/home/Hao/Work/Cmts/tfidf_bow.npy')

tfidf_term = np.load('/home/Hao/Work/Cmts/tfidf_term.npy')
tfidf_list = np.load('/home/Hao/Work/Cmts/tfidf_list.npy')

fv_mid = []
for ele in tfidf_list:
    tmp = np.load('/media/Hao/My Book/mid_total_fv/' + ele + '.npy')
    fv_mid.append(tmp)
fv_mid = np.array(fv_mid)
fv_hmdb_train = np.load('/home/Hao/Work/hmdb_0.4_fv.npy')
fv_hmdb_test = np.load('/home/Hao/Work/hmdb_testing_fv.npy')

#test = get_greedy_group([0, 0])
#print test
#ID = range(842)
#seeds = range(842)
#groups = np.load('/home/Hao/Work/Cmts/greedy_group2.npz')
groups = np.load('/home/Hao/Work/Cmts/group_level_expan_only_too_small2.npz')
#groups = groups['group']
level_end = np.load('/home/Hao/Work/Cmts/level_selection2.npz')
input_list = []
#test_set = np.load('/home/Hao/Work/mid_testing_set.npy')
items = range(len(groups['ID']))
#items = range(len(test_set))
#random.shuffle(items)
for i in items:
    if len(groups['group'][i]) > 1:
        input_list.append([groups['ID'][i], groups['group'][i]])
        print input_list[-1]
#print oracle(input_list[0])
#p = Pool(8)
#ALL = p.map(level_set, input_list)
#ALL = p.map(oracle, input_list)
#print level_expan(input_list[0])
#p = Pool(8)
#ALL = p.map(level_expan, input_list)
#print level_set(input_list[0])
#test_2methods(input_list[0])
p = Pool(4)
ALL = p.map(test_2methods, input_list)
#ALL = p.map(get_greedy_group, input_list)
#ALL = p.map(group_analysis, input_list)
#[ID, group, acc_mean, acc_min, acc_25] = p.map(get_greedy_group, input_list)
#np.savez('greedy_group', ID = ID, group = group, acc_mean = acc_mean, acc_min = acc_min, acc_25 = acc_25)
#test = compute_term_acc(clu_group[0])
#np.save('group_acc', test)
