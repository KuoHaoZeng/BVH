import numpy as np
import scipy.io as sio
import generate_group


M = 2
def get_D(M):
    package = np.load('/home/Hao/Work/Cmts/raw/total_training_mid_cal4_4_11_' + str(M) + '.npz')

    N = 0
    for i in xrange(len(package['label'])):
        if package['label'][i] == 1:
            N += 1

    model_list = []
    for i in xrange(5):
        C = 0.01 * 10 ** (i)
        model_list = '/home/Hao/Work/Cmts/calibrate/ranking/ranking_mid_cal_C' + str(C) + 'M' + str(M) + '.npy'

        rep = generate_group.pred_ranking_feature(model_list, package['fv'])

        Data = np.zeros([N, M])
        Y = -np.zeros([N, M])
        idx = 0
        for i in xrange(len(rep)):
            if package['label'][i] == 1:
                Data[idx] = rep[i]
                for j in xrange(M):
                    if package['cluster'][i] == j:
                        Y[idx][j] = 1
                    else:
                        Y[idx][j] = 0
                idx += 1
        sio.savemat('/home/Hao/Work/Cmts/calibrate/ranking/ranking_mid_cal_Data_C' + str(C) + 'M' + str(M) + '.mat', {'Data' : Data, 'Y' : Y})

for i in xrange(5):
    get_D(i + 1)
    print 'case ' + str(i + 1) + ' done.'
