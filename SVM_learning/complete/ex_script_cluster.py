import numpy as np
import generate_group, sys

def wraper(inp, n):
    test = np.load('/home/Hao/Work/Cmts/raw/raw_test_mid_cal_rep' + inp + '.npz')
    train = np.load('/home/Hao/Work/Cmts/raw/raw_train_mid_cal_rep' + inp + '.npz')
    #test = np.load('/home/Hao/Work/Cmts/raw/total_testing' + inp + '.npz')
    #train = np.load('/home/Hao/Work/Cmts/raw/total_training' + inp + '.npz')
    test_fv = np.load('/home/Hao/Work/Cmts/raw/total_testing' + inp + '.npz')
    train_fv = np.load('/home/Hao/Work/Cmts/raw/total_training' + inp + '.npz')

    #fv_train = np.float32(train['fv'])
    #print fv_train.shape
    center = generate_group.kmeans_train(train, n, 50, 300, 0.0001, 'cv2')
    print center
    labels = generate_group.kmeans_pred(train, center, 'cv2')


    #np.savez('/home/Hao/Work/Cmts/raw/raw_test_mid_cal_rep' + inp + '_' + str(n), fv = test['fv'], label = test['label'], q= test['q'])
    #np.savez('/home/Hao/Work/Cmts/raw/raw_train_mid_cal_rep' + inp + '_' + str(n), fv = train['fv'], label = train['label'], q= train['q'], cluster = labels)
    np.savez('/home/Hao/Work/Cmts/raw/total_testing_mid_cal' + inp + '_' + str(n), fv = test_fv['fv'], label = test_fv['label'], q= test_fv['q'])
    np.savez('/home/Hao/Work/Cmts/raw/total_training_mid_cal' + inp + '_' + str(n), fv = train_fv['fv'], label = train_fv['label'], q= train_fv['q'], cluster = labels)

for i in range(5):
    wraper('4_4_11', i + 1)
