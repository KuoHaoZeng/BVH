import numpy as np
from numpy import linalg
from scipy import ndimage, signal, stats
import scipy.spatial.distance as dis
import scipy.io as sio
from multiprocessing import Pool
from sklearn import cluster
import sys, math, copy, time, cv2, os, heapq, Vcontutil, Vcont, mlpy, random, sklearn.metrics, subprocess

def get_video_list(f):
        video_list = []
        for line in f:
            if len(line) > 1:
                temp = line[0 : -1]
                video_list.append(temp)
            #xx = temp.split('/')
            #video_names.append(xx[len(xx) - 1])
        return video_list

def get_tuple_index(arr):
        index = np.zeros(len(arr))
        for i in xrange(len(arr)):
                index[i] = arr[i][0]
        return index

def get_number_case(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x))]

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

def grous_extract(ID, groups, mAP):
    ## ID = id of each group
    ## groups = all groups
    ## mAP = mean average precision of each group
    ## return = extracted ID, extracted groups, extracted mAP

    # set overlapping threshold and initialition
    threshold = 0.7
    everybody = np.zeros(842)
    ID_extract = []
    groups_extract = []
    mAP_extract = []

    while 1:
        mAP = list(mAP)
        maximum = max(mAP)
        print maximum
        idx = mAP.index(maximum)
        ID_extract.append(ID[idx])
        groups_extract.append(groups[idx])
        mAP_extract.append(mAP[idx])

        num = len(ID)
        overlap = np.zeros(num)
        for i in xrange(num):
            tmp = 0
            for ele in groups[i]:
                if ele in groups_extract[-1]:
                #if everybody[ele] == 1:
                    tmp += 1.
            #overlap[i] = tmp / (len(groups[i]))
            overlap[i] = tmp / (len(groups[i]) + len(groups_extract[-1]) - tmp)
        delete_idx = np.where(overlap >= threshold)[0]
        #print delete_idx
        ID = np.delete(ID, delete_idx, axis = 0)
        groups = np.delete(groups, delete_idx, axis = 0)
        mAP = np.delete(mAP, delete_idx, axis = 0)
        #mAP = list(mAP)
        #if len(ID) < 1:
        #    break

        #maximum = max(mAP)
        #print maximum
        #idx = mAP.index(maximum)
        #ID_extract.append(ID[idx])
        #groups_extract.append(groups[idx])
        #mAP_extract.append(mAP[idx])
        for ele in groups_extract[-1]:
            everybody[ele] = 1
        if float(sum(everybody)) / 842 >= 0.9:
            break
    print len(ID_extract)
    return ID_extract, groups_extract, mAP_extract

def train_SVM_group(data):
    ## data = [ID, group(ID)]
    ## return [ID, 3-folder cross-validation mean average precision]
    ID = data[0]
    group = list(data[1])
    print 'seed ID: ' + str(ID)

    ## remove testing data from training set
    tmp = list(group)
    for ele in tmp:
        if ele in test_set:
            group.remove(ele)
    path = '/home/Hao/Work/one_features/'
    if os.path.isfile(path + str(ID) + '_' + str(len(group)) + '.model'):
        print 'done'
        #return
    if len(group) == 0:
        print group
        return group

    ## get negative set expan by adding term accuracy = 0 video
    group_top_terms = get_tfidf(tfidf_bow, group, 10)
    group_top_terms = get_tuple_index(group_top_terms)
    neg_expan = []
    acc_list = []
    for i in xrange(842):
        if i not in group:
            query_top_terms = get_tfidf(tfidf_bow, [i], 10)
            query_top_terms = get_tuple_index(query_top_terms)
            if i not in test_set:
                acc_list.append(acc(query_top_terms, group_top_terms))
            if acc_list[-1] == 0 and i not in test_set:
                neg_expan.append(i)
    calibrate_case = heapq.nlargest(40, enumerate(acc_list), key=lambda x:x[1])
    calibrate_case = get_tuple_index(calibrate_case)
    print 'negative data expan size: ' + str(len(neg_expan))
    #return ID, calibrate_case

    ## prepare fv & label for training linear SVM classifier
    pos = np.delete(range(842), group)
    fv_pos = np.delete(fv_mid, pos, axis = 0)
    neg = np.delete(range(842), neg_expan)
    fv_neg_mid = np.delete(fv_mid, neg, axis = 0)
    fv_neg_hmdb = fv_hmdb_train
    if len(neg_expan) == 0:
        fv_neg = fv_neg_hmdb
        print 'seed ID: ' + str(ID) + ' has no mid negative examples!!'
    else:
        fv_neg = fv_neg_mid
    fv_neg = np.vstack((fv_neg_mid, fv_neg_hmdb))
    label_pos = np.ones(len(fv_pos))
    label_neg = np.zeros(len(fv_neg))
    fv = np.vstack((fv_pos, fv_neg))
    label = np.hstack((label_pos, label_neg))

    ## train & save model
    svm = Vcontutil.linearSVM_T(fv, label, 10, {0 : 1, 1 : 1})
    svm.save_model(path + str(ID) + '_' + str(len(group)) + '.model')
    sys.stdout.flush()

def pred_mid_feature(model_path_list, test_data):
    SVM = []
    for ele in model_path_list:
        SVM_tmp = mlpy.LibLinear.load_model(ele)
        SVM.append(SVM_tmp._w())
    SVM = np.array(SVM)
    bias = np.ones(len(test_data))
    bias.shape = 1, -1
    print bias.shape
    delete_idx = np.delete(range(842), test_data)
    fv = np.delete(fv_mid, delete_idx, axis = 0)
    fv = fv.T
    print fv.shape
    fv = np.vstack((fv, bias))
    y = np.dot(SVM, fv)
    y = y.T
    return y

def pred_raw_feature(model_path_list, fv):
    SVM = []
    for ele in model_path_list:
        SVM_tmp = mlpy.LibLinear.load_model(ele)
        SVM.append(SVM_tmp._w())
    SVM = np.array(SVM)
    bias = np.ones(len(fv))
    bias.shape = 1, -1
    print bias.shape
    fv = fv.T
    print fv.shape
    fv_new = np.vstack((fv, bias))
    y = np.dot(SVM, fv_new)
    y = y.T
    fv = fv.T
    return y

def pred_ranking_feature(model_path, fv):
    SVM = []
    SVM_tmp = np.load(model_path)
    for ele in SVM_tmp:
        SVM.append(ele)
    SVM = np.array(SVM)
    fv = fv.T
    print fv.shape
    y = np.dot(SVM, fv)
    y = y.T
    fv = fv.T
    return y

def mid_eva(model_list, test_data, scores):
    video = test_data
    query_top_terms = get_tfidf(tfidf_bow, [video], 10)
    query_top_terms = get_tuple_index(query_top_terms)
    scores = list(scores)
    maximum = max(scores)
    scores_sort = sorted(scores)
    group = []
    for i in xrange(1):
        #idx = scores.ind`ex(maximum)
        idx = scores.index(scores_sort[-1 - i])
        group += model_list[idx][1]
    group_top_terms = get_tfidf(tfidf_bow, group, 10)
    group_top_terms = get_tuple_index(group_top_terms)
    return acc(query_top_terms, group_top_terms)

def Distance(test, train, func):
    test_num = len(test)
    train_num = len(train)
    Dis = np.zeros([test_num, train_num])
    nn = []
    for i in xrange(test_num):
        for j in xrange(train_num):
            Dis[i][j] = -func(test[i],train[j])
            if i == j:
                Dis[i][j] = -sys.maxint
            #Dis[i][j] = np.linalg.norm(test[i] - train[j])
            #print Dis[i][j]
        nn.append(heapq.nlargest(40, enumerate(Dis[i]), key=lambda x:x[1]))
        print nn[-1]
    return nn

def e_distance(a, b):
    return np.linalg.norm(a - b)

def calibrate_prepare(model_path, group):
    print model_path
    # positive data
    pos = np.delete(range(842), group)
    fv_pos = np.delete(fv_mid, pos, axis = 0)
    label_pos = np.ones(len(fv_pos))
    # negative data
    fv_neg = fv_hmdb_test
    label_neg = np.zeros(len(fv_neg))

    # stack & bias term
    fv = np.vstack((fv_pos, fv_neg))
    bias = np.ones(len(fv))
    bias.shape = 1, -1
    fv = fv.T
    fv = np.vstack((fv, bias))
    label = np.hstack((label_pos, label_neg))

    # SVM evaluation
    SVM = mlpy.LibLinear.load_model(model_path)
    SVM = SVM._w()
    y = np.dot(SVM, fv)
    y.shape = -1, fv.shape[1]
    print y.shape, label.shape
    return y, label

def calibrate_mapping(param, scores):
    scores_map = np.zeros(scores.shape)
    for i in xrange(len(scores)):
        for j in xrange(len(param)):
            scores_map[i][j] = 1 / (1 + math.exp(-param[j][0]*(scores[i][j] - param[j][1])))
            print scores[i][j], scores_map[i][j]
    return scores_map

def kmeans_train(D, n, ini_iter, max_iter, eps, func):
    N = 0
    for ele in D['label']:
        if ele == 1:
            N +=1
    data = np.zeros([N, D['fv'].shape[1]], dtype = np.float32)
    q = np.zeros(N)
    xx = 0
    for i in xrange(len(D['fv'])):
        if D['label'][i] == 1:
            data[xx] = D['fv'][i]
            q[xx] = D['q'][i]
            xx += 1
    '''
    q_case = get_number_case(q)
    if len(q_case) > 10:
        idx = np.delete(range(len(q_case)), random.sample(range(len(q_case)), 10))
        q_case = np.delete(q_case, idx)
        x_new = []
        for i in xrange(len(data)):
            if q in q_case:
                x_new.append(data[i])
        D = np.array(x_new)
    print D.shape
    sys.exit()
    '''
    if func == 'cv2':
        data = np.array(data, dtype = np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
        ret, label, center = cv2.kmeans(data, n, criteria, ini_iter, cv2.KMEANS_RANDOM_CENTERS)
        center = [center, label]
    elif func == 'skl':
        center = cluster.KMeans(n, 'k-means++', ini_iter, max_iter, eps, 'auto', 1)
        center.fit(data)
    return center

def kmeans_pred(D, center, func):
    q_case = get_number_case(D['q'])
    n_case = len(q_case)
    Label = np.zeros(len(D['q']))
    if func == 'cv2':
        labels = center[1]
        idx = 0
        for i in xrange(len(D['q'])):
            print 'case: ' + str(i)
            if D['label'][i] == 1:
                Label[i] = labels[idx]
                idx += 1
            else:
                Label[i] = -1
    elif func == 'skl':
        for i in xrange(len(D['q'])):
            print 'case: ' + str(i)
            if D['label'][i] == 1:
                Label[i] = center.predict(D['fv'][i])
            else:
                Label[i] = -1
    return Label
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

if not os.path.isfile('/home/Hao/Work/Cmts/fv_mid_all.npy'):
    fv_mid = []
    for ele in tfidf_list:
        tmp = np.load('/media/Hao/My Book/mid_total_fv/' + ele + '.npy')
        fv_mid.append(tmp)
    fv_mid = np.array(fv_mid)
    np.save('/home/Hao/Work/Cmts/fv_mid_all', fv_mid)
else:
    fv_mid = np.load('/home/Hao/Work/Cmts/fv_mid_all.npy')
fv_hmdb_train = np.load('/home/Hao/Work/hmdb_0.4_fv.npy')
fv_hmdb_test = np.load('/home/Hao/Work/hmdb_testing_fv.npy')

fv_raw_test_integral = np.load('/media/Hao/My Book/raw_total_fv/total_testing2_4_7.npz')
fv_raw_test = fv_raw_test_integral['fv']
fv_raw_train_integral = np.load('/media/Hao/My Book/raw_total_fv/total_training2_4_7.npz')
fv_raw_train = fv_raw_train_integral['fv']
#test = get_greedy_group([0, 0])
#print test
#ID = range(842)
#seeds = range(842)
#groups = np.load('/home/Hao/Work/Cmts/greedy_group2.npz')
'''
groups = np.load('/home/Hao/Work/Cmts/group_level_expan_remove_too_small2.npz')
in_groups = list(groups['group'])
groups = np.load('/home/Hao/Work/Cmts/group_level_expan_only_too_small2.npz')
for ele in groups['group']:
    if len(ele) > 1:
        in_groups.append(ele)
in_ID = []
in_mAP = []
mAPs = np.load('/home/Hao/Work/Cmts/mAP_4_3.npy')
for ele in mAPs:
    in_ID.append(ele[0])
    in_mAP.append(ele[1])
mAPs = np.load('/home/Hao/Work/Cmts/mAP_4_4.npy')
for ele in mAPs:
    in_ID.append(ele[0])
    in_mAP.append(ele[1])
xx=0
yy=[]
for ele in xrange(len(in_mAP)):
    if in_mAP[ele] == 0:
        yy.append(len(in_groups[ele]))
        xx += 1
print xx, np.mean(yy)
ALL = grous_extract(in_ID, in_groups, in_mAP)
'''
#groups = groups['group']
#level_end = np.load('/home/Hao/Work/Cmts/level_selection2.npz')
#input_list = []
test_set = np.load('/home/Hao/Work/mid_testing_set.npy')
train_set = np.load('/home/Hao/Work/mid_training_set.npy')
'''
groups = np.load('/home/Hao/Work/Cmts/final_group_4_5.npz')
calibrate_case = np.load('/home/Hao/Work/Cmts/final_one_not_pos&neg_4_8.npz')
test_scores_integral = np.load('/home/Hao/Work/Cmts/raw/raw_test_mid_mixed_rep.npz')
test_scores = test_scores_integral['fv']
train_scores_integral = np.load('/home/Hao/Work/Cmts/raw/raw_train_mid_mixed_rep.npz')
train_scores = train_scores_integral['fv']
par = sio.loadmat('/home/Hao/Work/Cmts/calibrate/par_4_6.mat')
par = par['par']
par = sio.loadmat('/home/Hao/Work/Cmts/calibrate/par_one_4_9.mat')
par_one = par['par']
'''
#ALL = calibrate_mapping(par, test_scores)
#test_scores = np.load('/home/sunmin/smax_test_rep.npy')
#train_scores = np.load('/home/sunmin/smax_train_rep.npy')
#train_scores = np.delete(fv_mid, test_set, axis = 0)
#test_scores = np.delete(fv_mid, train_set, axis = 0)
#test_scores = np.load('/home/Hao/Work/Cmts/calibrate/test_mid_cal_rep.npy')
#train_scores = np.load('/home/Hao/Work/Cmts/calibrate/train_mid_cal_rep.npy')
#ALL = Distance(train_scores, train_scores, dis.cosine)
'''
path = '/home/Hao/Work/mid_features/'
path_one = '/home/Hao/Work/one_features/'

ALL = []
for i in xrange(len(calibrate_case['ID'])):
    ALL.append(calibrate_prepare(path_one + str(calibrate_case['ID'][i]) + '_' + \
                                 #str(len(groups['group'][i])) + '.model', \
                                 str(1) + '.model', \
                                 calibrate_case['group'][i][0:10]))
'''
input_list = []
'''
for i in xrange(len(groups['ID'])):
#for i in xrange(842):
    #if i not in test_set:
        #input_list.append([i, calibrate_case['ID'][i], calibrate_case['group'][i][0:10]])
        input_list.append(path + str(groups['ID'][i]) + '_' + \
                      str(len(groups['group'][i])) + '.model')
        #input_list.append(path + str(i) + '_1.model')
        print input_list[-1]

for i in xrange(842):
    if i not in test_set:
        input_list.append(path_one + str(i) + '_' + str(1) + '.model')
        print input_list[-1]
#print calibrate_prepare(input_list[0])
#p = Pool(4)
#ALL = p.map(calibrate_prepare, input_list)
#ALL = pred_mid_feature(input_list, train_set)
ALL = pred_raw_feature(input_list, fv_raw_train)
'''
#ALL = []
'''
for i in xrange(len(groups['ID'])):
    input_list.append([groups['ID'][i], groups['group'][i]])
    print input_list[-1]
'''
'''
for i in xrange(842):
    if i in test_set:
        if not os.path.isfile('/home/Hao/Work/one_features/' + str(i) + '_1.model'):
            input_list.append([i, [i]])
            print input_list[-1]
test_set = []
#train_SVM_group(input_list[0])
'''
'''
for i in xrange(len(train_set)):
    input_list.append([train_set[i], [train_set[i]]])
    print input_list[-1]
#ALL = []
#for i in xrange((len(test_set))):
#    ALL.append(mid_eva(input_list, test_set[i], test_scores[i]))
'''
#ALL = pred_mid_feature(input_list, test_set)
#train_SVM_group(input_list[0])
#p = Pool(4)
#ALL = p.map(train_SVM_group, input_list)
#items = range(len(groups['ID']))
#items = range(len(test_set))
#random.shuffle(items)
#for i in items:
#    input_list.append([groups['ID'][i], groups['group'][i]])
#    print input_list[-1]
#print oracle(input_list[0])
#p = Pool(8)
#ALL = p.map(level_set, input_list)
#ALL = p.map(oracle, input_list)
#print level_expan(input_list[0])
#p = Pool(8)
#ALL = p.map(level_expan, input_list)
#print level_set(input_list[0])
#test_2methods(input_list[0])
#p = Pool(4)
#ALL = p.map(test_2methods, input_list)
#ALL = p.map(get_greedy_group, input_list)
#ALL = p.map(group_analysis, input_list)
#[ID, group, acc_mean, acc_min, acc_25] = p.map(get_greedy_group, input_list)
#np.savez('greedy_group', ID = ID, group = group, acc_mean = acc_mean, acc_min = acc_min, acc_25 = acc_25)
#test = compute_term_acc(clu_group[0])
#np.save('group_acc', test)
