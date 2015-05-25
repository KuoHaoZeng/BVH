from multiprocessing import Pool
import json, mid

List = open('/media/al-farabi/Seagate Backup Plus Drive/UW/HL/surfing/test_v1/vlist_msun_sel.json').read()
List = json.loads(List)

input_list = []
'''
for i in List:
    for j in i[0]:
        input_list.append('/media/al-farabi/Seagate Backup Plus Drive/UW/HL/surfing/' + j[32:43] + '/' + j[32:43] + '.mp4')
print len(input_list)
'''
List = open('/media/al-farabi/Seagate Backup Plus Drive/UW/HL/skating2/test_v1/vlist_msun_sel.json').read()
List = json.loads(List)

for i in List:
    for j in i[0]:
        input_list.append('/media/al-farabi/Seagate Backup Plus Drive/UW/HL/skating2/' + j[17:28] + '/' + j[17:28] + '.mp4')

p = Pool(4)
p.map(mid.mid_features, input_list)
