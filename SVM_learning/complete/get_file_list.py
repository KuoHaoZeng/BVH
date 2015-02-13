import os, sys, subprocess
import pickle as pk
#path = '/home/al-farabi/Desktop/video_pool/hmdb51_org/'
path = '/home/Hao/Work/fv/'
#target_path = '/home/al-farabi/Desktop/hmdb_features_fix360/'
#target_path = '/home/Hao/Work/mid_features_fix360/'
dirs = os.listdir(path)
f = open('/home/Hao/Work/mid_list.txt', 'r')
f2 = open('/home/Hao/Work/mid_list2.txt', 'w')


def get_cmtz(sel_file):
        cmtz = []
        for ele in sel_file:
                a = ele[2].find('all')
                b = ele[2].find('bow')
                cmtz.append('_' + ele[2][b + 4 : a])
        return cmtz

file_path = '/home/Hao/Work/viral_data/mid_cmts/bow/sel/selK5_T5.pkl'
sel_file = pk.load(open(file_path,'r'))
cmtz = get_cmtz(sel_file)
'''
dirs2 = os.listdir(target_path)
for i in dirs2:
    f.write(target_path + i + '\n')
    print i
sys.exit()
'''
def SuForC(Path):
    Re = Path.replace(';','\;')
    Re = Re.replace('(','\(')
    Re = Re.replace(')','\)')
    Re = Re.replace('&','\&')
    return Re

video_names = []
def get_video_list(f, video_names):
    for line in f:
        temp = line[0 : len(line) - 1]
        #video_list.append(temp)
        xx = temp.split('/')
        video_names.append(xx[len(xx) - 1] + '.npy')
    return video_names
video_names = get_video_list(f, video_names)

count = 0
for i in cmtz:
    folder = path + i
    if i[len(i) - 4 : len(i)] == '.mp4' or i[len(i) - 3 : len(i)] == '.py' or i[len(i) - 4 : len(i)] == '.txt':
        continue
    if i + '.npy' in dirs:
        count += 1
        print count
        f2.write(folder + '.npy\n')
	#subprocess.call('rm ' + i, shell = True);
    '''
    files = os.listdir(folder)
    for j in files:
        if j[0 : 5] == 'jukin':
            continue
        if j[len(j) - 4 : len(j)] == '.mp4':
            f.write(target_path + j[0 : len(j) - 4] + '\n')
            print j
        #else:
        #   temp = SuForC(j)
        #   subprocess.call('rm ' + path + i + '/' + temp, shell=True)
    '''
#f.close()
