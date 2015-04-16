import generate_group, sys
import numpy as np
'''
f = open('/home/Hao/Work/hmdb_list.txt', 'r')
video_list = generate_group.get_video_list(f)
f.close()
train_set = np.load('/home/Hao/Work/Cmts/hmdb_training_set.npy')
f = open('/home/Hao/Work/Cmts/group_fig/hmdb_path.txt', 'w')
for i in train_set:
    f.write(video_list[i][18:-4] + '\n')
'''

groups = np.load('/home/Hao/Work/Cmts/final_group_4_5.npz')
tfidf_term = np.load('/home/Hao/Work/Cmts/tfidf_term.npy')
tfidf_bow = np.load('/home/Hao/Work/Cmts/tfidf_bow.npy')

group = groups['group']

f = open('/home/Hao/Work/Cmts/group_fig/group_term.txt', 'w')
for i in range(len(group)):
    print 'group ID: ' + str(groups['ID'][i])
    f.write('group ID: ' + str(groups['ID'][i]) + '\n')
    for ele in group[i]:
        #print video_list[ele][18:-4]
        f.write(video_list[ele][18:-4] + ' ')
    f.write('\n')
    top_term = generate_group.get_tfidf(tfidf_bow, group[i], 10)
    top_term = generate_group.get_tuple_index(top_term)
    for ele in top_term:
        f.write(tfidf_term[ele] + ' ')
    f.write('\n')
