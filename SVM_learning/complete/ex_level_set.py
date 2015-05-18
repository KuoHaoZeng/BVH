import numpy as np
from multiprocessing import Pool
import generate_group, sys

group = np.load('/home/Hao/Work/Cmts/greedy_group2.npz')
input_list = []
for i in xrange(len(group['ID'])):
    input_list.append([group['ID'][i], group['acc_mean'][i]])

#p = Pool(8)
#p.map(generate_group.level_set, input_list)
ALL = []
for i in range(len(input_list)):
    print i
    ALL.append(generate_group.level_set(input_list[i]))
