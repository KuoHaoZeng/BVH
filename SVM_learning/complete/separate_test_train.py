import numpy as np
import sys

def get_video_list(f, video_list):
        for line in f:
                temp = line[0 : len(line) - 1]
                video_list.append(temp)
                #xx = temp.split('/')
                #video_names.append(xx[len(xx) - 1])
        return video_list

f = open('/home/hao/Desktop/match_list.txt', 'r')
video_list = []
video_list = get_video_list(f, video_list)
input_list = []
for i in video_list:
	input_list.append('/media/hao/My Book/raw_total_fv/' + i[32:] + '.npz')

test = np.load('/home/hao/Desktop/raw_testing_set.npy')
train = np.load('/home/hao/Desktop/raw_training_set.npy')
training_fv = np.delete(input_list, test)
testing_fv = np.delete(input_list, train)

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

np.savez('/media/hao/My Book/raw_total_fv/total_training', fv = fv_total, label = label_total, q = q_total)

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

np.savez('/media/hao/My Book/raw_total_fv/total_testing', fv = fv_total, label = label_total, q = q_total)
