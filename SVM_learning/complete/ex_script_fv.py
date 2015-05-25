from multiprocessing import Pool
import json, mid, os

List = open('/media/Hao/Seagate Backup Plus Drive/UW/HL/surfing/test_v1/vlist_msun_sel.json').read()
List = json.loads(List)

N = 139
Q = 1000
idx = 0

input_list = []
for i in xrange(len(List)):
    for j in List[i][0]:
        input_list.append(['surfing', j[32:43], Q])
        Q += 1

List = open('/media/Hao/Seagate Backup Plus Drive/UW/HL/skating2/test_v1/vlist_msun_sel.json').read()
List = json.loads(List)

for i in xrange(len(List)):
    for j in List[i][0]:
        input_list.append(['skating2', j[17:28], Q])
        Q += 1
'''
#print input_list[-1]
for ele in input_list:
    #if not os.path.isfile('/media/Hao/Seagate Backup Plus Drive/HL_fv/' + ele[1] + '.npz'):
    if not os.path.isfile('/media/Hao/My Book/raw_whole_fv_demo/' + ele[1] + '.npz'):
        mid.fisherGN_rank_UW(ele)
    #mid.fisherGN_rank_UW(ele)
'''
#print mid.fisherGN_rank_UW(input_list[0])
p = Pool(4)
ALL = p.map(mid.fisherGN_rank_UW, input_list)
