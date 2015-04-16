import numpy as np
import scipy.io as sio
import generate_group, os

def wraper(inp):
    fv_raw_test_integral = np.load('/home/Hao/Work/Cmts/raw/total_testing' + inp + '.npz')
    fv_raw_test = fv_raw_test_integral['fv']
    fv_raw_train_integral = np.load('/home/Hao/Work/Cmts/raw/total_training' + inp + '.npz')
    fv_raw_train = fv_raw_train_integral['fv']

    # Maybe bug.
    test = np.load('/home/Hao/Work/Cmts/raw/raw_testing_set' + inp + '.npy')
    train = np.load('/home/Hao/Work/Cmts/raw/raw_training_set' + inp + '.npy')
    test_mid = np.load('/home/Hao/Work/mid_testing_set.npy')

    path = '/home/Hao/Work/mid_features/'
    path_one = '/home/Hao/Work/one_features/'
    groups = np.load('/home/Hao/Work/Cmts/final_group_4_5.npz')
    input_list = []
    for i in xrange(len(groups['ID'])):
            input_list.append(path + str(groups['ID'][i]) + '_' + \
                          str(len(groups['group'][i])) + '.model')
            print input_list[-1]

    raw_mid_test_group = generate_group.pred_raw_feature(input_list, fv_raw_test)
    raw_mid_train_group = generate_group.pred_raw_feature(input_list, fv_raw_train)
    np.savez('/home/Hao/Work/Cmts/raw/raw_test_mid_rep' + inp, fv = raw_mid_test_group, label = fv_raw_test_integral['label'], q = fv_raw_test_integral['q'])
    np.savez('/home/Hao/Work/Cmts/raw/raw_train_mid_rep' + inp, fv = raw_mid_train_group, label = fv_raw_train_integral['label'], q = fv_raw_train_integral['q'])

    par = sio.loadmat('/home/Hao/Work/Cmts/calibrate/par_4_6.mat')
    par = par['par']
    raw_mid_test_group_cal = generate_group.calibrate_mapping(par, raw_mid_test_group)
    raw_mid_train_group_cal = generate_group.calibrate_mapping(par, raw_mid_train_group)
    np.savez('/home/Hao/Work/Cmts/raw/raw_test_mid_cal_rep' + inp, fv = raw_mid_test_group_cal, label = fv_raw_test_integral['label'], q = fv_raw_test_integral['q'])
    np.savez('/home/Hao/Work/Cmts/raw/raw_train_mid_cal_rep' + inp, fv = raw_mid_train_group_cal, label = fv_raw_train_integral['label'], q = fv_raw_train_integral['q'])

    input_list = []
    for i in xrange(842):
        if i not in test_mid:
            input_list.append(path_one + str(i) + '_' + str(1) + '.model')
            print input_list[-1]
    raw_mid_test_one = generate_group.pred_raw_feature(input_list, fv_raw_test)
    raw_mid_train_one = generate_group.pred_raw_feature(input_list, fv_raw_train)
    par_one = sio.loadmat('/home/Hao/Work/Cmts/calibrate/par_one_4_9.mat')
    par_one = par_one['par']
    raw_mid_test_one_cal = generate_group.calibrate_mapping(par, raw_mid_test_one)
    raw_mid_train_one_cal = generate_group.calibrate_mapping(par, raw_mid_train_one)

    raw_mid_test_group = raw_mid_test_group.T
    raw_mid_train_group = raw_mid_train_group.T
    raw_mid_test_one = raw_mid_test_one.T
    raw_mid_train_one = raw_mid_train_one.T
    fv_raw_test = fv_raw_test.T
    fv_raw_train = fv_raw_train.T

    raw_mid_test_stack = np.vstack((fv_raw_test, raw_mid_test_group))
    raw_mid_train_stack = np.vstack((fv_raw_train, raw_mid_train_group))
    raw_mid_test_stack_mix = np.vstack((raw_mid_test_stack, raw_mid_test_one))
    raw_mid_train_stack_mix = np.vstack((raw_mid_train_stack, raw_mid_train_one))

    raw_mid_test_stack = raw_mid_test_stack.T
    raw_mid_train_stack = raw_mid_train_stack.T
    np.savez('/home/Hao/Work/Cmts/raw/raw_test_mid_stack_rep' + inp, fv = raw_mid_test_stack, label = fv_raw_test_integral['label'], q = fv_raw_test_integral['q'])
    np.savez('/home/Hao/Work/Cmts/raw/raw_train_mid_stack_rep' + inp, fv = raw_mid_train_stack, label = fv_raw_train_integral['label'], q = fv_raw_train_integral['q'])

    raw_mid_test_stack_mix = raw_mid_test_stack_mix.T
    raw_mid_train_stack_mix = raw_mid_train_stack_mix.T
    np.savez('/home/Hao/Work/Cmts/raw/raw_test_mid_stack_mixed_rep' + inp, fv = raw_mid_test_stack_mix, label = fv_raw_test_integral['label'], q = fv_raw_test_integral['q'])
    np.savez('/home/Hao/Work/Cmts/raw/raw_train_mid_stack_mixed_rep' + inp, fv = raw_mid_train_stack_mix, label = fv_raw_train_integral['label'], q = fv_raw_train_integral['q'])

    raw_mid_test = np.vstack((raw_mid_test_group, raw_mid_test_one))
    raw_mid_train = np.vstack((raw_mid_train_group, raw_mid_train_one))
    raw_mid_test = raw_mid_test.T
    raw_mid_train = raw_mid_train.T
    np.savez('/home/Hao/Work/Cmts/raw/raw_test_mid_mixed_rep' + inp, fv = raw_mid_test, label = fv_raw_test_integral['label'], q = fv_raw_test_integral['q'])
    np.savez('/home/Hao/Work/Cmts/raw/raw_train_mid_mixed_rep' + inp, fv = raw_mid_train, label = fv_raw_train_integral['label'], q = fv_raw_train_integral['q'])

    raw_mid_test_group_cal = raw_mid_test_group_cal.T
    raw_mid_train_group_cal = raw_mid_train_group_cal.T
    raw_mid_test_one_cal = raw_mid_test_one_cal.T
    raw_mid_train_one_cal = raw_mid_train_one_cal.T

    raw_mid_test_cal = np.vstack((raw_mid_test_group_cal, raw_mid_test_one_cal))
    raw_mid_train_cal = np.vstack((raw_mid_train_group_cal, raw_mid_train_one_cal))
    raw_mid_test_cal = raw_mid_test_cal.T
    raw_mid_train_cal = raw_mid_train_cal.T
    np.savez('/home/Hao/Work/Cmts/raw/raw_test_mid_mixed_cal_rep' + inp, fv = raw_mid_test_cal, label = fv_raw_test_integral['label'], q = fv_raw_test_integral['q'])
    np.savez('/home/Hao/Work/Cmts/raw/raw_train_mid_mixed_cal_rep' + inp, fv = raw_mid_train_cal, label = fv_raw_train_integral['label'], q = fv_raw_train_integral['q'])


wraper('5_4_14')
