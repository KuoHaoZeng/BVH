import generate_group
import numpy as np
import scipy.spatial.distance as dis

fv_mid = np.load('/home/Hao/Work/Cmts/fv_mid_all.npy')
test_set = np.load('/home/Hao/Work/mid_testing_set.npy')
train_set = np.load('/home/Hao/Work/mid_training_set.npy')
#test = np.load('/home/Hao/Work/mid_testing_fv.npy')
#train = np.load('/home/Hao/Work/mid_training_fv.npy')
test = np.delete(fv_mid, train_set, axis = 0)
train = np.delete(fv_mid, test_set, axis = 0)

#ALL = generate_group.Distance(test, train, dis.cosine)
