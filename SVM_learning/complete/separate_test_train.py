import numpy as np
import sys

def get_video_list(f, video_list):
        for line in f:
                temp = line[0 : len(line) - 1]
                video_list.append(temp)
                #xx = temp.split('/')
                #video_names.append(xx[len(xx) - 1])
        return video_list

def separate(inp):
    f = open('/home/Hao/Work/Cmts/match_list_backup.txt', 'r')
    video_list = []
    video_list = get_video_list(f, video_list)
    input_list = []
    for i in video_list:
        input_list.append('/media/Hao/My Book/raw_total_fv/' + i + '.npz')

    test = np.load('/home/Hao/Work/Cmts/raw/raw_testing_set' + inp + '.npy')
    train = np.load('/home/Hao/Work/Cmts/raw/raw_training_set' + inp + '.npy')
    test_idx = np.delete(range(len(input_list)), test)
    testing_fv = np.delete(input_list, test_idx)
    train_idx = np.delete(range(len(input_list)), train)
    training_fv = np.delete(input_list, train_idx)

    Fir = True
    for i in xrange(len(training_fv)):
        print 'case ' + str(i)
        tmp = np.load(training_fv[i])
        fv = tmp['fv']
        label = tmp['label']
        q = tmp['q']
        if Fir == True:
            fv_total = fv
            label_total = label
            q_total = q
            Fir = False
        else:
            fv_total = np.vstack((fv_total, fv))
            label_total = np.hstack((label_total, label))
            q_total = np.hstack((q_total, q))

    #np.savez('/media/Hao/My Book/raw_total_fv/total_training' + inp, fv = fv_total, label = label_total, q = q_total)
    np.savez('/home/Hao/Work/Cmts/raw/debug/total_training' + inp, fv = fv_total, label = label_total, q = q_total)

    Fir = True
    for i in xrange(len(testing_fv)):
        print 'case ' + str(i)
        tmp = np.load(testing_fv[i])
        fv = tmp['fv']
        label = tmp['label']
        q = tmp['q']
        if Fir == True:
            fv_total = fv
            label_total = label
            q_total = q
            Fir = False
        else:
            fv_total = np.vstack((fv_total, fv))
            label_total = np.hstack((label_total, label))
            q_total = np.hstack((q_total, q))

    #np.savez('/media/Hao/My Book/raw_total_fv/total_testing' + inp, fv = fv_total, label = label_total, q = q_total)
    np.savez('/home/Hao/Work/Cmts/raw/debug/total_testing' + inp, fv = fv_total, label = label_total, q = q_total)

separate('4_4_11')
