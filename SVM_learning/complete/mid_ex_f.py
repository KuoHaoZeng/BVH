import Vcontutil, os, subprocess, sys
import pickle as pk
from multiprocessing import Pool

f=open('/home/al-farabi/Desktop/manual.txt','r')
video_list = []
inlist = []
crop = []
for line in f:
	temp = line[0 : len(line) - 1]
	video_list.append(temp)
	xx = temp.split(' ')
	inlist.append(xx[0])
	if len(xx)>3 and '*' in xx[3]:
		crop.append(xx[0])

file_path = '/home/al-farabi/Desktop/selK5_T5.pkl'
sel_file = pk.load(open(file_path,'r'))
cmtz = []
for ele in sel_file:
	a = ele[2].find('all')
	b = ele[2].find('bow')
	cmtz.append('_' + ele[2][b + 4 : a])

folder = '/home/al-farabi/Desktop/mid/'
dirs = os.listdir(folder)
output_dir = '/home/al-farabi/Desktop/mid_features'

def ffmpeg_duration(target):
	FFMPEG_BIN = 'ffmpeg'
	command = [FFMPEG_BIN,'-i', target, '-']
	pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	pipe.stdout.readline()
	pipe.terminate()
	infos = pipe.stderr.read()
	index = infos.find('Duration:')
	duration = int(infos[index + 13 : index + 15]) * 60 + int(infos[index + 16 : index + 18])
	return duration

def ffmpeg_duration(target, output, start, end):
	subprocess.call('sudo ffmpeg -i ' + target + ' -ss ' + start + ' -t ' + end + ' -vcodec copy -acodec copy ' + output, shell = True)

def video_edit(text):
	temp = text.split(' ')
	ele = temp[0]
	target = folder + ele + '/' + ele + '.mp4'
        jukin = folder + ele + '/jukin' + ele + '.mp4'
	
	if os.path.isfile(jukin):
		return
	
	start = temp[1]
	end = temp[2]
	if not os.path.isfile(jukin):
		subprocess.call('sudo mv ' + target + ' ' + jukin, shell = True)
	
	ffmpeg_duration(jukin, target, start, end)
	print 'done!!'

def mid_features(text, f = None):
	temp = text.split(' ')
        ele = temp[0]
        target = folder + ele + '/' + ele + '.mp4'
	Vcontutil.Extracting(target, output_dir)

p = Pool(3)
p.map(mid_features, video_list)
