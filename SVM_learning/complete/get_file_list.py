import os, sys, subprocess
path = '/home/al-farabi/Desktop/video_pool/hmdb51_org/'
dirs = os.listdir(path)
#f = open('hmdb_list.txt', 'w')

def SuForC(Path):
    Re = Path.replace(';','\;')
    Re = Re.replace('(','\(')
    Re = Re.replace(')','\)')
    Re = Re.replace('&','\&')
    return Re

for i in dirs:
    folder = path + i + '/'
    files = os.listdir(folder)
    for j in files:
        if j[len(j) - 4 : len(j)] == '.avi':
            #f.write(path + i + '/' + j[0 : len(j) - 4] + '\n')
            print 'ok!'
        else:
            temp = SuForC(j)
            subprocess.call('rm ' + path + i + '/' + temp, shell=True)
#f.close()
