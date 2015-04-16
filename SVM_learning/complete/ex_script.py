import generate_group, sys
import numpy as np

f = open('/home/Hao/Work/raw_check_list.txt', 'r')
video_list = generate_group.get_video_list(f)
video = []
for ele in video_list:
    xx = ele.split('\t')
    if len(xx[2]) == 1 and int(xx[2]) == 1:
        video.append(int(xx[1]))


