import numpy as np
import generate_group, sys, time

test = np.load('/home/Hao/Work/Cmts/raw/raw_test_mid_rep6_4_20.npz')
train = np.load('/home/Hao/Work/Cmts/raw/raw_train_mid_rep6_4_20.npz')
#test = np.load('/home/Hao/Work/Cmts/raw/total_testing6_4_20.npz')
#train = np.load('/home/Hao/Work/Cmts/raw/total_training6_4_20.npz')
test_fv = np.load('/home/Hao/Work/Cmts/raw/total_testing6_4_20.npz')
train_fv = np.load('/home/Hao/Work/Cmts/raw/total_training6_4_20.npz')
def wraper(inp, n):
    #test = np.load('/home/Hao/Work/Cmts/raw/total_testing' + inp + '.npz')
    #train = np.load('/home/Hao/Work/Cmts/raw/total_training' + inp + '.npz')

    #fv_train = np.float32(train['fv'])
    #print fv_train.shape
    s = time.time()
    center = generate_group.kmeans_train(train, n, 1, 10, 1, 'skl')
    #print center
    labels = generate_group.kmeans_pred(train, center, 'skl')
    print 'time cose: ' + str(time.time() - s)


    np.savez('/home/Hao/Work/Cmts/raw/total_testing_mid' + inp + '_' + str(n) + '_13', fv = test_fv['fv'], label = test_fv['label'], q= test_fv['q'])
    np.savez('/home/Hao/Work/Cmts/raw/total_training_mid' + inp + '_' + str(n) + '_13', fv = train_fv['fv'], label = train_fv['label'], q= train_fv['q'], cluster = labels)

for i in range(5):
    #s = time.time()
    wraper('6_4_20', i + 1)
    #print 'time cose: ' + str(time.time() - s)
