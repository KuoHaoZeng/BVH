import os, sys, subprocess
#path = '/home/al-farabi/Desktop/video_pool/hmdb51_org/'
path = '/home/al-farabi/Desktop/mid/'
#target_path = '/home/al-farabi/Desktop/hmdb_features_fix360/'
target_path = '/home/al-farabi/Desktop/mid_features_fix360/'
dirs = os.listdir(path)
f = open('/home/al-farabi/Desktop/mid_list.txt', 'w')

dirs2 = os.listdir(target_path)
for i in dirs2:
    f.write(target_path + i + '\n')
    print i
sys.exit()

def SuForC(Path):
    Re = Path.replace(';','\;')
    Re = Re.replace('(','\(')
    Re = Re.replace(')','\)')
    Re = Re.replace('&','\&')
    return Re

for i in dirs:
    folder = path + i + '/'
    if i[len(i) - 4 : len(i)] == '.mp4' or i[len(i) - 3 : len(i)] == '.py' or i[len(i) - 4 : len(i)] == '.txt':
        continue
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
f.close()
